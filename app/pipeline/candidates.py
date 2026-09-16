"""
The candidate pool: every style code any signal proposed, keyed by
normalized code, with the evidence sources that proposed it and whatever
catalogue metadata we can attach (StockX, our DB).
"""
import logging
from typing import Dict, Iterable, List, Optional

from app.extraction.normalize import parse_title
from app.extraction.sku_detector import normalize_style_code
from app.schemas import EvidenceSource, ScoredCandidate, StockXProduct
from app.stockx.catalog import StockXCatalog, choose_product
from app.stockx.client import StockXBudgetExhausted, StockXRequestFailed

logger = logging.getLogger(__name__)


class CandidatePool:
    def __init__(self):
        self._by_code: Dict[str, ScoredCandidate] = {}

    def __len__(self) -> int:
        return len(self._by_code)

    def get(self, code: str) -> Optional[ScoredCandidate]:
        return self._by_code.get(normalize_style_code(code))

    def all(self) -> List[ScoredCandidate]:
        return list(self._by_code.values())

    def add(self, code: str, source: EvidenceSource, display: str = "", **meta) -> Optional[ScoredCandidate]:
        norm = normalize_style_code(code)
        if not norm:
            return None
        c = self._by_code.get(norm)
        if c is None:
            c = ScoredCandidate(style_code=norm, style_code_display=display or code)
            self._by_code[norm] = c
        if display and (not c.style_code_display or c.style_code_display == c.style_code):
            c.style_code_display = display
        if not any(s.kind == source.kind and s.detail == source.detail for s in c.sources):
            c.sources.append(source)
        for key, val in meta.items():
            if val in ("", None):
                continue
            if getattr(c, key, None) in ("", None):
                setattr(c, key, val)
        return c

    def add_stockx_product(self, product: StockXProduct, source: EvidenceSource) -> List[ScoredCandidate]:
        """A compound styleId ("A/B") yields one candidate per segment, all
        pointing at the same StockX product."""
        out = []
        for seg in product.style_codes or [normalize_style_code(product.style_id)]:
            c = self.add(seg, source, display=_display_segment(product.style_id, seg))
            if c is not None:
                self.apply_stockx_metadata(c, product)
                out.append(c)
        return out

    @staticmethod
    def apply_stockx_metadata(c: ScoredCandidate, product: StockXProduct) -> None:
        parsed = parse_title(product.title, product.brand)
        c.stockx_product_id = c.stockx_product_id or product.product_id
        c.stockx_url = c.stockx_url or product.url
        c.name = product.title or c.name
        c.brand = parsed.brand or product.brand or c.brand
        c.model = c.model or parsed.model
        c.colorway = product.colorway or c.colorway
        c.gender = (product.gender or c.gender or "").lower()
        if parsed.size_category != "unknown":
            c.size_category = parsed.size_category
        elif c.gender in ("child", "kids", "youth", "preschool", "toddler", "infant"):
            c.size_category = {"preschool": "ps", "toddler": "td", "infant": "td"}.get(c.gender, "gs")
        elif c.gender in ("men", "women", "unisex"):
            c.size_category = c.size_category or "adult"
        c.release_date = product.release_date or c.release_date
        c.retail_price = product.retail_price if product.retail_price is not None else c.retail_price
        if product.product_type:
            c.component_notes["product_type"] = product.product_type

    def enrich_from_stockx(self, catalog: StockXCatalog, max_lookups: int = 8,
                           priority: Optional[Iterable[str]] = None) -> List[str]:
        """Resolves candidates lacking StockX metadata by style code.
        Returns error strings; never raises."""
        errors: List[str] = []
        if not catalog.available:
            return ["stockx_not_configured"]
        order = list(priority or []) + [c.style_code for c in self._by_code.values()]
        seen, lookups = set(), 0
        for code in order:
            norm = normalize_style_code(code)
            c = self._by_code.get(norm)
            if c is None or norm in seen or c.stockx_product_id:
                continue
            seen.add(norm)
            if lookups >= max_lookups:
                c.component_notes["stockx"] = "lookup skipped (budget)"
                continue
            lookups += 1
            try:
                products = catalog.by_style_code(c.style_code_display or c.style_code)
            except (StockXRequestFailed, StockXBudgetExhausted) as exc:
                errors.append(f"{norm}: {exc}")
                continue
            except Exception as exc:
                errors.append(f"{norm}: {type(exc).__name__}")
                continue
            if not products:
                c.component_notes["stockx"] = "no catalog product for this style code"
                continue
            best, ambiguous = choose_product(products)
            if len(products) > 1:
                c.component_notes["stockx"] = (f"{len(products)} catalog listings share this style code"
                                               + (" and disagree" if ambiguous else " (duplicates, consistent)"))
                if ambiguous:
                    c.contradictions.append("stockx_multiple_products")
            self.apply_stockx_metadata(c, best)
        return errors


def _display_segment(style_id: str, norm_seg: str) -> str:
    for seg in (style_id or "").split("/"):
        cleaned = seg.strip()
        if normalize_style_code(cleaned) == norm_seg:
            import re
            return re.sub(r"\([^)]*\)", "", cleaned).strip()
    return norm_seg
