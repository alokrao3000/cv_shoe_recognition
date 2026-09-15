from typing import List, Optional

from pydantic import BaseModel


class Candidate(BaseModel):
    """One nearest-neighbor match from the reference index."""
    sku: str
    name: Optional[str] = None
    similarity: float


class MarketData(BaseModel):
    """StockX market snapshot for the identified product (see
    app/stockx_client.py::get_market)."""
    product_id: str
    url: str
    lowest_ask: Optional[float] = None
    highest_bid: Optional[float] = None
    currency: str = "USD"
    # Per-variant (size) breakdown, when available.
    sizes: List["SizeMarket"] = []


class SizeMarket(BaseModel):
    size: str
    lowest_ask: Optional[float] = None
    highest_bid: Optional[float] = None


class ReverseSearchMatch(BaseModel):
    """A SKU recovered via app/reverse_image_search.py when the local index
    had no confident match — separate from Candidate because it never went
    through the embedding similarity gate; StockX confirming the catalog
    match (a market lookup succeeding) is what stands in for that here."""
    sku: str
    source: str    # 'best_guess' or the URL the SKU was scraped from


class IdentifyResponse(BaseModel):
    identified: bool
    reason: Optional[str] = None          # set when identified=False: "no_confident_match" | "ambiguous_top_match"
    top_match: Optional[Candidate] = None
    candidates: List[Candidate] = []
    market: Optional[MarketData] = None
    market_error: Optional[str] = None    # set when identified but the StockX lookup itself failed
    reverse_search_match: Optional[ReverseSearchMatch] = None
    reverse_search_status: Optional[str] = None  # set when the fallback ran but found nothing usable:
                                                  # "blocked:<reason>" | "no_sku_found" | "sku_found_no_market:<sku>:<err>"


MarketData.model_rebuild()
