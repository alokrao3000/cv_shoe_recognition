"""
Transparent identity-match scoring, contradiction detection and confidence
tiers. Every number here is traceable: each component records the inputs it
compared, and every penalty names the contradiction.

    score = Σ weight_i × component_i          (weights from Settings)
    then contradictions subtract / reject, then verification adjusts,
    then the tier is read off the final score — with extra requirements for
    HIGH so that no single weak signal can produce a confident answer.
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.config import settings
from app.extraction.normalize import (
    ParsedTitle, canonical_model, category_compatible, colorway_similarity, colorway_tokens, gender_compatible,
    model_similarity, parse_title,
)
from app.extraction.sku_detector import normalize_style_code
from app.schemas import (
    CandidateVerdict, ProductPageData, ScoredCandidate, StyleCodeCandidate, VerificationResult, VisionAnalysis,
)
from app.search.web_discovery import WebCandidate

NEUTRAL = 0.5

# Failure codes (spec §20)
NO_PRODUCT_IMAGE = "NO_PRODUCT_IMAGE"
NO_METADATA = "NO_METADATA"
NO_SKU = "NO_SKU"
VISION_UNCERTAIN = "VISION_UNCERTAIN"
VISION_UNAVAILABLE = "VISION_UNAVAILABLE"
MULTIPLE_CANDIDATES = "MULTIPLE_CANDIDATES"
WEB_SEARCH_FAILED = "WEB_SEARCH_FAILED"
STOCKX_API_FAILED = "STOCKX_API_FAILED"
STOCKX_MATCH_UNCERTAIN = "STOCKX_MATCH_UNCERTAIN"
CONTRADICTORY_EVIDENCE = "CONTRADICTORY_EVIDENCE"
UNRESOLVED = "UNRESOLVED"
NOT_A_SNEAKER = "NOT_A_SNEAKER"
NO_CANDIDATES = "NO_CANDIDATES"
scoring_MULTIPLE = MULTIPLE_CANDIDATES


@dataclass
class Observation:
    """Everything we know about the query, from all inputs."""
    page: Optional[ProductPageData] = None
    vision: Optional[VisionAnalysis] = None
    title: str = ""
    page_codes: List[StyleCodeCandidate] = field(default_factory=list)
    web_codes: Dict[str, WebCandidate] = field(default_factory=dict)
    parsed_title: Optional[ParsedTitle] = None

    def __post_init__(self):
        t = self.title or (self.page.title if self.page else "")
        self.title = t
        if t and self.parsed_title is None:
            self.parsed_title = parse_title(t, self.page.brand if self.page else "")

    # ── derived query attributes (vision first, then page text) ──
    @property
    def brand(self) -> str:
        if self.vision and self.vision.brand and self.vision.brand_confidence >= 0.5:
            return self.vision.brand
        return (self.parsed_title.brand if self.parsed_title else "") or (self.vision.brand if self.vision else "")

    @property
    def model(self) -> str:
        if self.vision and self.vision.model and self.vision.model_confidence >= 0.5:
            return self.vision.model
        return (self.parsed_title.model if self.parsed_title else "") or (self.vision.model if self.vision else "")

    @property
    def colorway_texts(self) -> List[str]:
        out = []
        if self.vision:
            out += [self.vision.colorway, self.vision.official_colorway_guess]
        if self.page:
            out.append(self.page.raw_identifiers.get("color", ""))
        if self.parsed_title:
            out.append(self.parsed_title.colorway_text)
        return [c for c in out if c]

    @property
    def gender(self) -> str:
        for g in ((self.parsed_title.gender if self.parsed_title else ""), (self.vision.gender if self.vision else "")):
            if g and g not in ("unknown",):
                return g
        return "unknown"

    @property
    def size_category(self) -> str:
        for c in ((self.parsed_title.size_category if self.parsed_title else ""),
                  (self.vision.size_category if self.vision else "")):
            if c and c != "unknown":
                return c
        return "unknown"

    def _from_vision_only(self, attr: str) -> bool:
        """True when the query attribute comes from the vision model alone (no page/title evidence)."""
        if not self.vision:
            return False
        page_val = getattr(self.parsed_title, attr, "") if self.parsed_title else ""
        return not page_val or page_val == "unknown"

    def vision_is_certain(self, attr: str) -> bool:
        conf = {"model": self.vision.model_confidence, "brand": self.vision.brand_confidence}.get(attr, 0.0) if self.vision else 0.0
        return conf >= 0.9

    @property
    def price(self) -> Optional[float]:
        return self.page.price if self.page else None

    @property
    def labeled_page_codes(self) -> List[str]:
        return [c.code for c in self.page_codes if c.labeled]

    @property
    def visible_tag_code(self) -> str:
        return normalize_style_code(self.vision.visible_style_code) if self.vision and self.vision.visible_style_code else ""


def weights() -> Dict[str, float]:
    return {
        "sku_match": settings.weight_sku_match,
        "model_match": settings.weight_model_match,
        "colorway_match": settings.weight_colorway_match,
        "embedding_similarity": settings.weight_embedding_similarity,
        "metadata_match": settings.weight_metadata_match,
        "external_agreement": settings.weight_external_agreement,
    }


# ── components ──

def _sku_component(c: ScoredCandidate, obs: Observation) -> Tuple[float, str]:
    best, why = 0.0, "no direct style-code evidence"
    for s in c.sources:
        if s.kind == "page_sku":
            val = 1.0 if s.weight >= 0.85 else 0.8
            note = "style code printed on the product page" + (" (labeled)" if s.weight >= 0.85 else "")
        elif s.kind == "vision_tag":
            val, note = 0.9, "style code legible on tag/box in the photo"
        elif s.kind == "web":
            wc = obs.web_codes.get(c.style_code)
            agreement = wc.agreement if wc else 1
            trust = wc.trust if wc else 0.5
            val = min(1.0, 0.35 + 0.15 * agreement + 0.1 * min(trust, 2.0))
            note = f"web search: {agreement} domain(s) agree (trust {trust:.1f})"
        elif s.kind == "vision_guess":
            val, note = 0.35, "style code recalled by the vision model (unverified)"
        elif s.kind == "stockx_search":
            val, note = 0.15, "StockX catalog text search hit only"
        elif s.kind == "embedding":
            val, note = 0.1, "visual neighbour only"
        else:
            val, note = 0.0, s.kind
        if val > best:
            best, why = val, note
    return best, why


def _model_component(c: ScoredCandidate, obs: Observation) -> Tuple[float, str]:
    qm = obs.model
    cm = c.model or (parse_title(c.name).model if c.name else "")
    if not qm or not cm:
        return NEUTRAL, f"model unknown on one side (query={qm!r}, candidate={cm!r})"
    sim = model_similarity(qm, cm)
    return (sim if sim is not None else NEUTRAL), f"{qm!r} vs {cm!r}"


def _colorway_component(c: ScoredCandidate, obs: Observation) -> Tuple[float, str]:
    if not c.colorway and not c.name:
        return NEUTRAL, "candidate colorway unknown"
    best, why = None, "no comparable colorway text"
    # The catalogue colorway is authoritative; a colorway parsed out of the
    # product name is a lossy fallback ("Off White Black" → white/black).
    targets = [c.colorway] if c.colorway else ([parse_title(c.name).colorway_text] if c.name else [])
    for q in obs.colorway_texts:
        for t in targets:
            sim = colorway_similarity(q, t)
            if sim is not None and (best is None or sim > best):
                best, why = sim, f"{q!r} vs {t!r}"
    nick = (obs.vision.official_colorway_guess if obs.vision else "").strip().lower()
    if nick and c.name:
        nick_words = {w for w in re.split(r"[^a-z0-9]+", nick) if len(w) >= 2}
        cand_words = {w for w in re.split(r"[^a-z0-9]+", parse_title(c.name).colorway_text.lower()) if len(w) >= 2}
        if nick_words and nick_words == cand_words and (best is None or best < 0.95):
            best, why = 0.95, f"nickname {nick!r} is the candidate's colorway name"
        elif nick_words and nick_words < cand_words:
            # "Big Bubble England" inside "Big Bubble England SE" is fine; "Panda" inside
            # "Reverse Panda" is a different shoe — a multi-word nickname is specific, one word is not.
            val = 0.85 if len(nick_words) >= 2 else 0.5
            if best is None or best < val:
                best, why = val, f"nickname {nick!r} appears in candidate colorway {sorted(cand_words)}"
    if best is None:
        return NEUTRAL, why
    return best, why


def _embedding_component(c: ScoredCandidate) -> Tuple[float, str]:
    if c.embedding_similarity is None:
        return NEUTRAL, "no reference image to compare against"
    floor, ceil = settings.embedding_floor, settings.embedding_ceiling
    val = max(0.0, min(1.0, (c.embedding_similarity - floor) / max(1e-6, ceil - floor)))
    return val, f"cosine {c.embedding_similarity:.3f} (floor {floor}, ceiling {ceil})"


def _metadata_component(c: ScoredCandidate, obs: Observation) -> Tuple[float, str]:
    checks, notes = [], []
    if obs.price and c.retail_price:
        ratio = obs.price / c.retail_price
        ok = 0.55 <= ratio <= 1.35
        checks.append(1.0 if ok else 0.3)
        notes.append(f"price ratio {ratio:.2f}")
    g = gender_compatible(obs.gender, c.gender)
    if g is not None:
        checks.append(1.0 if g else 0.0)
        notes.append(f"gender {obs.gender}/{c.gender}")
    cat = category_compatible(obs.size_category, c.size_category)
    if cat is not None:
        checks.append(1.0 if cat else 0.0)
        notes.append(f"size category {obs.size_category}/{c.size_category}")
    fq, fc = canonical_brand_family(obs.brand), canonical_brand_family(c.brand)
    if fq and fc:
        checks.append(1.0 if fq == fc else 0.0)
        notes.append(f"brand {obs.brand}/{c.brand}")
    if not checks:
        return NEUTRAL, "no comparable metadata"
    return sum(checks) / len(checks), "; ".join(notes)


def canonical_brand_family(b: str) -> str:
    """'' for brands we don't know — an unknown string is not evidence of anything."""
    from app.extraction.normalize import brand_family, known_brand
    return brand_family(b).lower() if known_brand(b) else ""


def _agreement_component(c: ScoredCandidate) -> Tuple[float, str]:
    kinds = {s.kind for s in c.sources if s.kind not in ("stockx_search",)}
    strong = {"page_sku", "vision_tag", "web", "embedding", "vision_guess", "user"} & kinds
    return min(1.0, len(strong) / 2.0), f"independent signals: {sorted(strong)}"


def score_candidate(c: ScoredCandidate, obs: Observation) -> ScoredCandidate:
    comps: Dict[str, Tuple[float, str]] = {
        "sku_match": _sku_component(c, obs),
        "model_match": _model_component(c, obs),
        "colorway_match": _colorway_component(c, obs),
        "embedding_similarity": _embedding_component(c),
        "metadata_match": _metadata_component(c, obs),
        "external_agreement": _agreement_component(c),
    }
    w = weights()
    c.components = {k: round(v[0], 4) for k, v in comps.items()}
    for k, v in comps.items():
        c.component_notes[k] = v[1]
    c.score = round(sum(w[k] * comps[k][0] for k in w), 4)
    return c


# ── contradictions ──

def detect_contradictions(c: ScoredCandidate, obs: Observation) -> ScoredCandidate:
    """Adds named contradictions; hard ones reject the candidate, soft ones
    subtract from the score. Idempotent."""
    hard: List[str] = []
    soft: List[Tuple[str, float]] = []

    ptype = c.component_notes.get("product_type", "").lower()
    if ptype and not any(t in ptype for t in ("sneaker", "shoe", "footwear", "boot", "slide", "sandal", "clog")):
        hard.append(f"stockx_product_type:{ptype}")

    # Vision-only attributes can be wrong; they penalize but only veto when the
    # page/title says the same thing (or vision is near-certain).
    cat = category_compatible(obs.size_category, c.size_category)
    if cat is False:
        msg = f"size_category_mismatch:{obs.size_category}!={c.size_category}"
        if obs._from_vision_only("size_category"):
            soft.append((msg + " (vision only)", 0.25))
        else:
            hard.append(msg)
    g = gender_compatible(obs.gender, c.gender)
    if g is False:
        soft.append((f"gender_mismatch:{obs.gender}!={c.gender}", 0.15))

    qm, cm = obs.model, c.model or (parse_title(c.name).model if c.name else "")
    if qm and cm:
        cq, cc = canonical_model(qm)[0], canonical_model(cm)[0]
        if cq and cc and cq != cc:
            sim = model_similarity(qm, cm) or 0.0
            if sim < 0.5:
                if obs._from_vision_only("model") and not obs.vision_is_certain("model"):
                    soft.append((f"model_mismatch:{cq}!={cc} (vision only)", 0.3))
                else:
                    hard.append(f"model_mismatch:{cq}!={cc}")
            elif sim < 0.9:
                soft.append((f"model_height_unspecified:{cq}~{cc}", 0.05))

    cw = c.components.get("colorway_match")
    if cw is not None and cw < 0.2 and obs.colorway_texts and c.colorway:
        soft.append((f"colorway_conflict:{obs.colorway_texts[0]}!={c.colorway}", 0.3))

    tag = obs.visible_tag_code
    if tag and tag != c.style_code:
        soft.append((f"tag_style_code_mismatch:{tag}", 0.4))

    labeled = obs.labeled_page_codes
    if labeled and c.style_code not in labeled:
        soft.append((f"page_style_code_mismatch:{labeled[0]}", 0.25))

    fq, fc = canonical_brand_family(obs.brand), canonical_brand_family(c.brand)
    if fq and fc and fq != fc:
        if obs._from_vision_only("brand") and not obs.vision_is_certain("brand"):
            soft.append((f"brand_mismatch:{obs.brand}!={c.brand} (vision only)", 0.3))
        else:
            hard.append(f"brand_mismatch:{obs.brand}!={c.brand}")

    if obs.parsed_title and obs.parsed_title.collab and c.name:
        if obs.parsed_title.collab.lower() not in c.name.lower():
            soft.append((f"collab_missing:{obs.parsed_title.collab}", 0.2))

    if obs.vision and not obs.vision.images_consistent:
        soft.append(("query_images_inconsistent", 0.1))

    if not (c.name or c.model or c.colorway):
        # Nothing known about this code (no catalogue/StockX/DB record): it can't
        # be compared, so it can't be contradicted either — don't let it win by default.
        soft.append(("no_catalog_identity", 0.3))

    existing = set(c.contradictions)
    for h in hard:
        if h not in existing:
            c.contradictions.append(h)
    penalty = 0.0
    for s, p in soft:
        if s not in existing:
            c.contradictions.append(s)
        penalty += p
    if "stockx_multiple_products" in c.contradictions:
        penalty += 0.05
    if hard:
        c.rejected = True
        c.rejection_reason = hard[0]
    c.score = round(max(0.0, c.score - penalty), 4)
    c.component_notes["contradiction_penalty"] = f"-{penalty:.2f}" if penalty else "0"
    return c


# ── verification & tiers ──

def apply_verification(candidates: List[ScoredCandidate], result: Optional[VerificationResult]) -> None:
    if result is None:
        return
    by_code = {v.style_code: v for v in result.verdicts}
    for c in candidates:
        v = by_code.get(c.style_code)
        if v is None:
            continue
        c.verification = v
        if v.verdict == "different":
            c.rejected = True
            c.rejection_reason = c.rejection_reason or "verifier:different"
            for x in v.contradictions:
                c.contradictions.append(f"verifier:{x}")
        elif v.verdict == "uncertain":
            c.score = round(min(c.score, settings.confidence_medium - 0.01), 4)
        elif v.verdict == "same":
            # A direct visual confirmation against the photos is worth more than a
            # text-only colorway comparison — but never enough on its own (max +0.10).
            bonus = 0.10 if v.confidence >= 0.85 else 0.05 if v.confidence >= 0.7 else 0.0
            c.score = round(min(1.0, c.score + bonus), 4)


def tier(score: Optional[float]) -> str:
    if score is None:
        return "unresolved"
    if score >= settings.confidence_high:
        return "high"
    if score >= settings.confidence_medium:
        return "medium"
    if score >= settings.confidence_low:
        return "low"
    return "unresolved"


@dataclass
class Resolution:
    winner: Optional[ScoredCandidate]
    status: str                       # high | medium | low | unresolved
    confidence: float
    failure_codes: List[str]
    notes: List[str]


def resolve(candidates: List[ScoredCandidate], obs: Observation, verification: Optional[VerificationResult],
            stockx_confirmed: bool, verifier_available: bool) -> Resolution:
    """Picks the winner and decides the tier, enforcing the HIGH-tier
    requirements and ambiguity margin."""
    failures: List[str] = []
    notes: List[str] = []
    alive = sorted([c for c in candidates if not c.rejected], key=lambda c: -c.score)
    if not alive:
        return Resolution(None, "unresolved", 0.0, [NO_CANDIDATES if not candidates else CONTRADICTORY_EVIDENCE], notes)
    identified = [c for c in alive if c.name or c.model or c.colorway]
    if not identified:
        notes.append("no surviving candidate has a catalogue identity to verify against")
        return Resolution(alive[0], "unresolved", min(alive[0].score, settings.confidence_low - 0.01),
                          [STOCKX_MATCH_UNCERTAIN], notes)
    alive = identified

    winner = alive[0]
    conf = winner.score
    if len(alive) > 1 and alive[1].score >= conf - settings.ambiguity_margin:
        runner = alive[1]
        verified_apart = (winner.verification and winner.verification.verdict == "same"
                          and (runner.verification is None or runner.verification.verdict != "same"))
        if not verified_apart:
            failures.append(MULTIPLE_CANDIDATES)
            notes.append(f"runner-up {runner.style_code_display} within {settings.ambiguity_margin} of the winner")
            conf = min(conf, settings.confidence_medium - 0.01)

    # Spec "Example A": a style code printed on the retailer page, confirmed by
    # StockX, with nothing contradicting it, is the strongest evidence there is.
    verified_same = winner.verification is not None and winner.verification.verdict == "same" \
        and winner.verification.confidence >= 0.7
    page_confirmed = (winner.style_code in obs.labeled_page_codes and stockx_confirmed and not winner.contradictions
                      and winner.components.get("model_match", NEUTRAL) >= NEUTRAL
                      # a text colorway mismatch is overruled by the verifier looking at the actual photos
                      and (winner.components.get("colorway_match", NEUTRAL) >= NEUTRAL or verified_same)
                      and scoring_MULTIPLE not in failures
                      # photos the verifier couldn't reconcile with the printed code → MEDIUM, "use but verify"
                      and not (winner.verification is not None and winner.verification.verdict == "uncertain"))
    if page_confirmed and conf < settings.confidence_high:
        notes.append("page-printed style code confirmed by StockX with no contradictions → HIGH")
        conf = settings.confidence_high

    status = tier(conf)
    if status == "high":
        verified_same = winner.verification is not None and winner.verification.verdict == "same"
        if not (verified_same or page_confirmed):
            status = "medium"
            conf = min(conf, settings.confidence_high - 0.01)
            notes.append("HIGH requires verifier agreement or a page-printed style code confirmed by StockX"
                         + ("" if verifier_available else " (verifier unavailable)"))
    if any(x.startswith("verifier:") for x in winner.contradictions) and status in ("high", "medium"):
        status = "low"
        conf = min(conf, settings.confidence_medium - 0.01)
    return Resolution(winner, status, round(conf, 4), failures, notes)
