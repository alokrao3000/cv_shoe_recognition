"""
Uses the web as a bridge to a style code: generate targeted queries from
what we know (retailer title, vision attributes, partial codes), search,
and harvest validated style codes from result titles/snippets, recording
which pages said so. Agreement across independent domains is evidence;
one random page is not.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.extraction import sku_detector
from app.extraction.normalize import parse_title
from app.schemas import ProductPageData, VisionAnalysis
from app.search.provider import ImageResult, SearchError, SearchProvider, WebResult

logger = logging.getLogger(__name__)

# Domains whose product pages reliably print the official style code.
_TRUSTED_DOMAINS: Dict[str, float] = {
    "stockx.com": 1.0, "goat.com": 1.0, "flightclub.com": 0.9, "stadiumgoods.com": 0.9,
    "nike.com": 1.0, "adidas.com": 1.0, "newbalance.com": 1.0, "asics.com": 1.0, "converse.com": 1.0,
    "vans.com": 1.0, "puma.com": 1.0, "reebok.com": 1.0, "salomon.com": 1.0, "on.com": 1.0, "hoka.com": 1.0,
    "sneakernews.com": 0.8, "solecollector.com": 0.8, "kicksonfire.com": 0.7, "sneakerfiles.com": 0.7,
    "hypebeast.com": 0.7, "nicekicks.com": 0.7, "footlocker.com": 0.8, "finishline.com": 0.8,
    "jdsports.com": 0.7, "champssports.com": 0.7, "hibbett.com": 0.7, "dtlr.com": 0.7, "shoepalace.com": 0.7,
    "sneakersnstuff.com": 0.7, "kith.com": 0.7, "undefeated.com": 0.7, "extrabutterny.com": 0.7,
    "ebay.com": 0.4, "grailed.com": 0.4, "amazon.com": 0.3, "walmart.com": 0.2, "poshmark.com": 0.3,
}
_DEFAULT_DOMAIN_TRUST = 0.5


@dataclass
class WebCandidate:
    code: str
    display: str
    brand_hint: str = ""
    sources: List[str] = field(default_factory=list)       # page URLs
    domains: List[str] = field(default_factory=list)
    titles: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    trust: float = 0.0                                      # sum of domain trust across distinct domains

    @property
    def agreement(self) -> int:
        return len(set(self.domains))


def build_queries(page: Optional[ProductPageData], vision: Optional[VisionAnalysis],
                  title: Optional[str] = None, partial_codes: Optional[List[str]] = None) -> List[str]:
    queries: List[str] = []

    def add(q: str):
        q = " ".join(q.split())
        if q and q.lower() not in {x.lower() for x in queries}:
            queries.append(q)

    raw_title = (page.title if page and page.title else title) or ""
    if raw_title:
        add(f"{raw_title} style code")
        parsed = parse_title(raw_title, page.brand if page else "")
        if parsed.model:
            add(f"{parsed.model} {parsed.colorway_text} style code".strip())
    if page and page.raw_identifiers.get("color") and raw_title:
        add(f"{raw_title} {page.raw_identifiers['color']}")
    if vision and vision.brand and vision.model:
        color = vision.official_colorway_guess or vision.colorway
        add(f"{vision.brand} {vision.model} {color} style code")
        if vision.sub_model:
            add(f"{vision.brand} {vision.model} {vision.sub_model} {color}")
        for name in vision.likely_release_names[:2]:
            add(f"{name} style code")
    for code in (partial_codes or [])[:3]:
        add(code)
    return queries[:6]


def _harvest(results: List[WebResult], query: str, brand_hint: Optional[str],
             pool: Dict[str, WebCandidate]) -> None:
    for r in results:
        text = f"{r.title} {r.snippet}"
        cands = sku_detector.detect_style_codes(text, f"web:{r.domain}", brand_hint, base_confidence=0.5)
        if not cands:
            continue
        # Only the best 2 codes per result — a listicle mentioning ten SKUs is weak evidence for any of them.
        for c in cands[:2]:
            wc = pool.get(c.code)
            if wc is None:
                wc = pool[c.code] = WebCandidate(code=c.code, display=c.display, brand_hint=c.brand_hint)
            if r.url and r.url not in wc.sources:
                wc.sources.append(r.url)
            if r.domain not in wc.domains:
                wc.domains.append(r.domain)
                wc.trust += _TRUSTED_DOMAINS.get(r.domain, _DEFAULT_DOMAIN_TRUST) * (1.0 if c.labeled else 0.7)
            if r.title and r.title not in wc.titles:
                wc.titles.append(r.title[:160])
            if query not in wc.queries:
                wc.queries.append(query)


def discover(queries: List[str], provider: SearchProvider, brand_hint: Optional[str] = None,
             num: int = 10) -> Tuple[List[WebCandidate], List[str]]:
    """Returns (candidates sorted by trust desc, error strings)."""
    pool: Dict[str, WebCandidate] = {}
    errors: List[str] = []
    if not provider.available:
        return [], ["no_provider"]
    for q in queries:
        try:
            results = provider.web_search(q, num=num)
        except SearchError as exc:
            errors.append(f"{q}: {exc}")
            logger.info(f"web search failed for {q!r}: {exc}")
            continue
        _harvest(results, q, brand_hint, pool)
    return sorted(pool.values(), key=lambda c: (-c.trust, -c.agreement)), errors


def find_reference_images(query: str, provider: SearchProvider, num: int = 6) -> List[ImageResult]:
    """Candidate reference photos for a known (name + style code). Prefers
    larger images and trusted domains; the caller downloads in order until
    one decodes."""
    if not provider.available:
        return []
    try:
        results = provider.image_search(query, num=num)
    except SearchError as exc:
        logger.info(f"image search failed for {query!r}: {exc}")
        return []

    def rank(r: ImageResult):
        big = 1 if r.width >= 400 and r.height >= 300 else 0
        return (-_TRUSTED_DOMAINS.get(r.domain, _DEFAULT_DOMAIN_TRUST), -big)

    return sorted(results, key=rank)
