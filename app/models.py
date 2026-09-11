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


class IdentifyResponse(BaseModel):
    identified: bool
    reason: Optional[str] = None          # set when identified=False: "no_confident_match" | "ambiguous_top_match"
    top_match: Optional[Candidate] = None
    candidates: List[Candidate] = []
    market: Optional[MarketData] = None
    market_error: Optional[str] = None    # set when identified but the StockX lookup itself failed


MarketData.model_rebuild()
