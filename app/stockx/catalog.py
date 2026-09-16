"""
StockX catalog access with a DB-backed TTL cache. StockX is the destination
(market data, canonical product), not the discovery layer — but its
catalog/search is also a cheap, official source of candidate releases with
style codes, so it's queried by name as well as by code.
"""
import logging
from datetime import timedelta
from typing import List, Optional, Tuple

from app.config import settings
from app.db.models import utcnow
from app.extraction.sku_detector import normalize_style_code
from app.schemas import MarketData, StockXProduct
from app.stockx import client as stockx_client

logger = logging.getLogger(__name__)

_MARKET_TTL = timedelta(hours=1)


def choose_product(products: List[StockXProduct]) -> Tuple[Optional[StockXProduct], bool]:
    """StockX sometimes lists the same style code twice (e.g. a legacy and a
    re-created listing, one with an empty colorway). Returns (best, ambiguous):
    ambiguous only when the listings disagree materially — different model,
    conflicting colorways, or different gender/size category."""
    if not products:
        return None, False
    if len(products) == 1:
        return products[0], False
    from app.extraction.normalize import colorway_similarity, model_similarity, parse_title
    base = products[0]
    ambiguous = False
    for p in products[1:]:
        ms = model_similarity(base.title, p.title)
        if ms is not None and ms < 0.9:
            ambiguous = True
        if base.colorway and p.colorway:
            cs = colorway_similarity(base.colorway, p.colorway)
            if cs is not None and cs < 0.9:
                ambiguous = True
        pb, pp = parse_title(base.title), parse_title(p.title)
        if pb.size_category != pp.size_category or (base.gender and p.gender and base.gender.lower() != p.gender.lower()):
            ambiguous = True

    def richness(p: StockXProduct) -> tuple:
        return (bool(p.colorway), bool(p.release_date), p.retail_price is not None, len(p.title))

    return max(products, key=richness), ambiguous


class StockXCatalog:
    def __init__(self, client: Optional[stockx_client.StockXAPIClient] = None, use_db_cache: bool = True):
        self._client = client
        self._use_db = use_db_cache
        self._mem: dict = {}

    @property
    def available(self) -> bool:
        return self._client is not None or stockx_client.is_configured()

    def _get_client(self) -> stockx_client.StockXAPIClient:
        if self._client is None:
            self._client = stockx_client.StockXAPIClient()
        return self._client

    # ── cache ──

    def _cache_get(self, key: str, ttl: timedelta):
        hit = self._mem.get(key)
        if hit and utcnow() - hit[0] < ttl:
            return hit[1]
        if not self._use_db:
            return None
        try:
            from app.db.models import StockXCache
            from app.db.session import db_session
            with db_session() as s:
                row = s.get(StockXCache, key)
                if row and utcnow() - row.fetched_at < ttl:
                    self._mem[key] = (row.fetched_at, row.payload)
                    return row.payload
        except Exception:
            logger.debug("stockx cache read failed (db unavailable?)", exc_info=True)
        return None

    def _cache_put(self, key: str, payload) -> None:
        self._mem[key] = (utcnow(), payload)
        if not self._use_db:
            return
        try:
            from app.db.models import StockXCache
            from app.db.session import db_session
            with db_session() as s:
                row = s.get(StockXCache, key)
                if row is None:
                    s.add(StockXCache(key=key, payload=payload, fetched_at=utcnow()))
                else:
                    row.payload, row.fetched_at = payload, utcnow()
        except Exception:
            logger.debug("stockx cache write failed (db unavailable?)", exc_info=True)

    # ── queries ──

    def search(self, query: str, page_size: int = 20) -> List[StockXProduct]:
        """Raises StockXRequestFailed / StockXBudgetExhausted on API failure."""
        q = " ".join((query or "").split())
        if not q:
            return []
        key = f"search:{q.lower()}:{page_size}"
        cached = self._cache_get(key, timedelta(hours=settings.stockx_cache_ttl_hours))
        if cached is not None:
            return [StockXProduct(**p) for p in cached]
        products = self._get_client().search_products(q, page_size=page_size)
        self._cache_put(key, [p.model_dump() for p in products])
        return products

    def by_style_code(self, style_code: str) -> List[StockXProduct]:
        """Products whose styleId contains this code as a segment — normally
        exactly one; more than one means StockX has split/duplicate listings
        and the caller must disambiguate."""
        norm = normalize_style_code(style_code)
        if not norm:
            return []
        key = f"style:{norm}"
        cached = self._cache_get(key, timedelta(hours=settings.stockx_cache_ttl_hours))
        if cached is not None:
            return [StockXProduct(**p) for p in cached]
        products = [p for p in self._get_client().search_products(style_code, page_size=20)
                    if norm in p.style_codes]
        self._cache_put(key, [p.model_dump() for p in products])
        return products

    def market(self, product: StockXProduct) -> Tuple[Optional[MarketData], Optional[str]]:
        key = f"market:{product.product_id}"
        cached = self._cache_get(key, _MARKET_TTL)
        if cached is not None:
            return MarketData(**cached), None
        market, err = self._get_client().market_for_product(product)
        if market is not None:
            self._cache_put(key, market.model_dump())
        return market, err

    def close(self):
        if self._client is not None:
            self._client.close()
