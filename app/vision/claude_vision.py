"""
Claude vision: (1) structured analysis of the query images — brand, model,
colorway, any legible style code, gender/size cues, distinctive features;
(2) verification of the top candidates against the same images, looking
specifically for evidence that a candidate is WRONG.

Both calls use structured outputs (client.messages.parse) so results are
schema-validated. The vision layer is one signal among several; it never
decides the match on its own (see app/pipeline/scoring.py).
"""
import base64
import io
import json
import logging
from typing import Dict, List, Optional

from PIL import Image
from pydantic import BaseModel

from app.config import settings
from app.schemas import CandidateVerdict, ScoredCandidate, VerificationResult, VisionAnalysis, VisionImageObservation

logger = logging.getLogger(__name__)


class VisionError(Exception):
    def __init__(self, code: str, detail: str = ""):
        self.code, self.detail = code, detail
        super().__init__(f"{code}: {detail}" if detail else code)


# ── Strict output schemas (every field required — structured outputs) ──

class _ImageObs(BaseModel):
    index: int
    view: str
    visible_text: List[str]
    depicts_same_product_as_first: bool


class _AnalysisOut(BaseModel):
    is_sneaker: bool
    brand: str
    model: str
    sub_model: str
    colorway: str
    official_colorway_guess: str
    visible_style_code: str
    gender: str
    size_category: str
    distinctive_features: List[str]
    likely_release_names: List[str]
    likely_style_codes: List[str]
    brand_confidence: float
    model_confidence: float
    colorway_confidence: float
    images: List[_ImageObs]
    images_consistent: bool
    notes: str


class _VerdictOut(BaseModel):
    style_code: str
    verdict: str
    confidence: float
    contradictions: List[str]
    reasoning: str


class _VerificationOut(BaseModel):
    verdicts: List[_VerdictOut]
    best_style_code: str
    notes: str


_ANALYSIS_SYSTEM = """You are an expert sneaker authenticator and product cataloguer. You will be shown one or more photos that are supposed to depict a single sneaker product (possibly from different angles, plus box/tag close-ups), sometimes with retailer text.

Your job is to describe exactly which release this is, as precisely as the evidence allows, and to be explicit about uncertainty.

Rules:
- brand: the brand as catalogued (use "Jordan" for Air Jordan products, "Nike" for other Nike incl. SB, "adidas", "New Balance", "ASICS", etc.).
- model: the silhouette including height where it matters ("Nike Dunk Low", "Air Jordan 1 High", "New Balance 9060", "adidas Samba OG"). Never omit Low/Mid/High when visible.
- sub_model: qualifiers like "Retro OG", "'07 LV8", "SB", "Premium", "Next Nature", collab name ("Travis Scott"), or "" if none.
- colorway: describe colours by placement in official-style order, e.g. "White/Black" (base/overlay), "Black/Fire Red-Cement Grey-Summit White". Lead with the base colour.
- official_colorway_guess: the nickname or official colorway name if you recognise it ("Panda", "Bred Reimagined", "Sea Salt"), else "".
- visible_style_code: ONLY a code you can actually read on a tag, box label, or in the retailer text. Never invent one. "" if not legible.
- gender: men | women | unisex | unknown — based on tag text, proportions, or explicit cues only.
- size_category: adult | gs | ps | td | unknown — GS/PS/TD only when a tag, box label, or obvious kids sizing shows it.
- distinctive_features: concrete visual facts (materials, panel colours, logo colours, midsole details, lace colours, tongue tags, heel tab text).
- likely_release_names: up to 5 specific releases this could be (most likely first), e.g. "Nike Dunk Low Retro White Black Panda".
- likely_style_codes: style codes you believe correspond to those releases from memory, most likely first. These are guesses to be verified externally — include them when you are reasonably sure, otherwise leave the list short.
- images: one entry per image, in order (index starting at 0). view is one of side, front, heel, top, outsole, box, tag, pair, other. visible_text: any legible text. depicts_same_product_as_first: false if this image shows a different shoe than image 0.
- images_consistent: false if any image contradicts another (different colorway/model).
- confidences are 0..1 and must reflect real uncertainty; do not report 0.9+ unless the evidence is unambiguous.
- If the images are not of a sneaker, set is_sneaker=false and leave other fields empty/0."""

_VERIFY_SYSTEM = """You are an expert sneaker authenticator acting as a skeptical verifier. You are shown the query photos of one sneaker product and a short list of candidate releases (with style codes and catalogue metadata, and sometimes a reference photo for a candidate).

For EACH candidate decide whether the query photos depict THAT EXACT RELEASE:
- "same": the visual evidence matches the candidate's model, height, colour placement, materials and any legible tag/box text, and nothing contradicts it.
- "different": there is at least one concrete contradiction (e.g. the query has a black swoosh but the candidate's colorway is white/grey; the query is a Low but the candidate is a High; a legible tag shows a different style code; a kids GS/PS/TD cue vs an adult release; a different collab or material).
- "uncertain": the photos genuinely cannot distinguish this candidate from another plausible one (e.g. near-identical colorways across years, or the differentiating area is not visible).

Rules:
- Name similarity is NOT evidence. Two candidates can have the same name and different colorways or sizes.
- List every contradiction you observe, concretely.
- confidence is 0..1 for the verdict you gave.
- best_style_code: the single candidate style code best supported, or "" if none is "same".
- Be willing to say "different" or "uncertain" for all candidates. A wrong "same" is far worse than an "uncertain"."""


def _prep_image(data: bytes, max_px: int) -> Optional[dict]:
    try:
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
    except Exception:
        return None
    w, h = img.size
    scale = max_px / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.standard_b64encode(buf.getvalue()).decode("ascii")}}


class VisionClient:
    def __init__(self, model: Optional[str] = None, client=None):
        self.model = model or settings.vision_model
        self._client = client

    @property
    def available(self) -> bool:
        if not settings.vision_enabled:
            return False
        if self._client is not None:
            return True
        import os
        return bool(settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
                    or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    def _get_client(self):
        if self._client is None:
            import anthropic
            kwargs = {}
            if settings.anthropic_api_key:
                kwargs["api_key"] = settings.anthropic_api_key
            self._client = anthropic.Anthropic(max_retries=2, timeout=120.0, **kwargs)
        return self._client

    def _parse(self, system: str, content: list, output_model, effort: str, max_tokens: int):
        import anthropic
        client = self._get_client()
        try:
            response = client.messages.parse(
                model=self.model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": content}],
                output_format=output_model,
                output_config={"effort": effort},
            )
        except anthropic.RateLimitError as exc:
            raise VisionError("rate_limited", str(exc)[:200]) from exc
        except anthropic.AuthenticationError as exc:
            raise VisionError("auth", str(exc)[:200]) from exc
        except anthropic.BadRequestError as exc:
            raise VisionError("bad_request", str(exc)[:300]) from exc
        except anthropic.APIStatusError as exc:
            raise VisionError(f"http_{exc.status_code}", str(exc)[:200]) from exc
        except anthropic.APIConnectionError as exc:
            raise VisionError("connection", str(exc)[:200]) from exc
        if response.stop_reason == "refusal":
            raise VisionError("refusal", getattr(getattr(response, "stop_details", None), "explanation", "") or "")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise VisionError("unparseable", f"stop_reason={response.stop_reason}")
        logger.info(f"[vision] {output_model.__name__} in={response.usage.input_tokens} "
                    f"cached={getattr(response.usage, 'cache_read_input_tokens', 0)} out={response.usage.output_tokens}")
        return response.parsed_output

    def analyze(self, images: List[bytes], context: Optional[Dict[str, str]] = None) -> VisionAnalysis:
        if not self.available:
            raise VisionError("not_configured")
        blocks = []
        for i, data in enumerate(images[: settings.max_images_per_request]):
            block = _prep_image(data, settings.vision_max_image_px)
            if block is None:
                continue
            blocks.append({"type": "text", "text": f"Image {i}:"})
            blocks.append(block)
        if not blocks:
            raise VisionError("no_decodable_images")
        ctx = {k: v for k, v in (context or {}).items() if v}
        prompt = "Analyse the sneaker in these images."
        if ctx:
            prompt += "\n\nRetailer context (may be incomplete or wrong — verify against the photos):\n" + \
                      json.dumps(ctx, ensure_ascii=False, indent=1)
        blocks.append({"type": "text", "text": prompt})
        out: _AnalysisOut = self._parse(_ANALYSIS_SYSTEM, blocks, _AnalysisOut, "medium", 4096)
        return VisionAnalysis(
            is_sneaker=out.is_sneaker, brand=out.brand, model=out.model, sub_model=out.sub_model,
            colorway=out.colorway, official_colorway_guess=out.official_colorway_guess,
            visible_style_code=out.visible_style_code, gender=(out.gender or "unknown").lower(),
            size_category=(out.size_category or "unknown").lower(),
            distinctive_features=out.distinctive_features, likely_release_names=out.likely_release_names,
            likely_style_codes=out.likely_style_codes, brand_confidence=out.brand_confidence,
            model_confidence=out.model_confidence, colorway_confidence=out.colorway_confidence,
            images=[VisionImageObservation(index=o.index, view=o.view, visible_text=o.visible_text,
                                           depicts_same_product_as_first=o.depicts_same_product_as_first)
                    for o in out.images],
            images_consistent=out.images_consistent, notes=out.notes,
        )

    def verify(self, images: List[bytes], candidates: List[ScoredCandidate],
               reference_images: Optional[Dict[str, bytes]] = None) -> VerificationResult:
        if not self.available:
            raise VisionError("not_configured")
        if not candidates:
            return VerificationResult()
        blocks = [{"type": "text", "text": "QUERY PHOTOS:"}]
        for i, data in enumerate(images[: settings.max_images_per_request]):
            block = _prep_image(data, settings.vision_max_image_px)
            if block:
                blocks.append({"type": "text", "text": f"Query image {i}:"})
                blocks.append(block)
        blocks.append({"type": "text", "text": "CANDIDATES:"})
        for c in candidates:
            meta = {
                "style_code": c.style_code_display or c.style_code, "name": c.name, "brand": c.brand,
                "model": c.model, "colorway": c.colorway, "gender": c.gender, "size_category": c.size_category,
                "release_date": c.release_date, "retail_price": c.retail_price,
            }
            blocks.append({"type": "text", "text": json.dumps({k: v for k, v in meta.items() if v}, ensure_ascii=False)})
            ref = (reference_images or {}).get(c.style_code)
            if ref:
                block = _prep_image(ref, 1024)
                if block:
                    blocks.append({"type": "text", "text": f"Reference photo for {meta['style_code']}:"})
                    blocks.append(block)
        blocks.append({"type": "text", "text": "Give a verdict for every candidate. Use the style_code exactly as given."})
        out: _VerificationOut = self._parse(_VERIFY_SYSTEM, blocks, _VerificationOut, "high", 4096)
        from app.extraction.sku_detector import normalize_style_code
        verdicts = [CandidateVerdict(style_code=normalize_style_code(v.style_code),
                                     verdict=v.verdict.lower().strip(), confidence=v.confidence,
                                     contradictions=v.contradictions, reasoning=v.reasoning)
                    for v in out.verdicts]
        return VerificationResult(verdicts=verdicts, best_style_code=normalize_style_code(out.best_style_code),
                                  notes=out.notes)
