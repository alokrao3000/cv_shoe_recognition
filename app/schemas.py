"""Pydantic models shared by the pipeline and the HTTP API."""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── StockX market data ──

class SizeMarket(BaseModel):
    size: str
    lowest_ask: Optional[float] = None
    highest_bid: Optional[float] = None


class MarketData(BaseModel):
    product_id: str
    url: str
    lowest_ask: Optional[float] = None
    highest_bid: Optional[float] = None
    currency: str = "USD"
    sizes: List[SizeMarket] = []


class StockXProduct(BaseModel):
    """A catalog/search hit, normalized."""
    product_id: str
    url_key: str = ""
    style_id: str = ""                      # raw, possibly compound ("A/B")
    style_codes: List[str] = []             # normalized segments
    title: str = ""
    brand: str = ""
    colorway: str = ""
    gender: str = ""
    release_date: str = ""
    retail_price: Optional[float] = None
    product_type: str = ""

    @property
    def url(self) -> str:
        return f"https://stockx.com/{self.url_key}" if self.url_key else ""


# ── Canonical product representation (spec §14) ──

class CanonicalProduct(BaseModel):
    brand: str = ""
    model: str = ""
    sub_model: str = ""
    colorway: str = ""
    style_code: str = ""
    gender: str = ""
    size_category: str = ""
    release_date: str = ""
    retailer_url: str = ""
    product_name: str = ""
    images: List[str] = []


# ── Extraction ──

class StyleCodeCandidate(BaseModel):
    code: str                               # normalized (A-Z0-9)
    display: str                            # as found
    source: str                             # page:jsonld | page:title | page:body | vision:tag | web:<domain> | stockx
    labeled: bool = False                   # preceded by "Style", "SKU", etc.
    brand_hint: str = ""
    confidence: float = 0.5


class ProductPageData(BaseModel):
    url: str = ""
    final_url: str = ""                      # after redirects (differs when a deleted product bounces to the store page)
    domain: str = ""
    title: str = ""
    brand: str = ""
    description: str = ""
    price: Optional[float] = None
    currency: str = ""
    images: List[str] = []
    breadcrumbs: List[str] = []
    category: str = ""
    sku_candidates: List[StyleCodeCandidate] = []
    raw_identifiers: Dict[str, str] = {}     # sku/mpn/gtin fields as found
    sizes: List[str] = []
    availability: str = ""
    extraction_sources: List[str] = []       # jsonld | shopify_json | next_data | opengraph | html
    error: Optional[str] = None


# ── Vision ──

class VisionImageObservation(BaseModel):
    index: int
    view: str = ""                          # side | front | heel | top | outsole | box | tag | other
    visible_text: List[str] = []
    depicts_same_product_as_first: bool = True


class VisionAnalysis(BaseModel):
    is_sneaker: bool = True
    brand: str = ""
    model: str = ""
    sub_model: str = ""
    colorway: str = ""
    official_colorway_guess: str = ""
    visible_style_code: str = ""
    gender: str = ""                        # men | women | unisex | unknown
    size_category: str = ""                 # adult | gs | ps | td | unknown
    distinctive_features: List[str] = []
    likely_release_names: List[str] = []
    likely_style_codes: List[str] = []
    brand_confidence: float = 0.0
    model_confidence: float = 0.0
    colorway_confidence: float = 0.0
    images: List[VisionImageObservation] = []
    images_consistent: bool = True
    notes: str = ""


class CandidateVerdict(BaseModel):
    style_code: str
    verdict: str                            # same | different | uncertain
    confidence: float = 0.0
    contradictions: List[str] = []
    reasoning: str = ""


class VerificationResult(BaseModel):
    verdicts: List[CandidateVerdict] = []
    best_style_code: str = ""
    notes: str = ""


# ── Candidates & scoring ──

class EvidenceSource(BaseModel):
    kind: str                               # page_sku | vision_tag | vision_guess | embedding | web | stockx | user
    detail: str = ""
    weight: float = 1.0


class ScoredCandidate(BaseModel):
    style_code: str
    style_code_display: str = ""
    name: str = ""
    brand: str = ""
    model: str = ""
    colorway: str = ""
    gender: str = ""
    size_category: str = ""
    release_date: str = ""
    retail_price: Optional[float] = None
    image_url: str = ""
    stockx_product_id: str = ""
    stockx_url: str = ""
    product_db_id: Optional[int] = None
    sources: List[EvidenceSource] = []
    embedding_similarity: Optional[float] = None
    components: Dict[str, float] = {}       # per-signal normalized scores 0..1
    component_notes: Dict[str, str] = {}
    score: float = 0.0
    contradictions: List[str] = []
    rejected: bool = False
    rejection_reason: str = ""
    verification: Optional[CandidateVerdict] = None


class StockXMatch(BaseModel):
    product_id: str
    url: str
    style_code: str
    product_name: str
    match_confidence: float
    lowest_ask: Optional[float] = None
    highest_bid: Optional[float] = None
    currency: str = "USD"
    sizes: List[SizeMarket] = []
    market_error: Optional[str] = None


class Stage(BaseModel):
    name: str
    label: str
    status: str = "pending"                 # pending | running | done | skipped | failed
    detail: str = ""
    ms: Optional[int] = None


# ── API ──

class IdentifyRequest(BaseModel):
    image_url: Optional[str] = None
    image_urls: List[str] = []
    product_url: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None


class IdentifyResult(BaseModel):
    id: str
    status: str
    identified: bool = False
    confidence: Optional[float] = None
    review_required: bool = False
    product: Optional[CanonicalProduct] = None
    stockx: Optional[StockXMatch] = None
    identification_method: List[str] = []
    candidates: List[ScoredCandidate] = []
    evidence: Dict[str, Any] = {}
    failure_codes: List[str] = []
    stages: List[Stage] = []
    input_images: List[str] = []
    input_url: Optional[str] = None
    input_title: Optional[str] = None
    created_at: Optional[datetime] = None
    error: Optional[str] = None


class ReviewDecisionRequest(BaseModel):
    action: str = Field(pattern="^(confirm|reject|select|manual_sku)$")
    style_code: Optional[str] = None
    notes: Optional[str] = None


class ReviewQueueItem(BaseModel):
    id: str
    status: str
    confidence: Optional[float]
    created_at: datetime
    input_title: Optional[str]
    input_url: Optional[str]
    input_images: List[str]
    top_candidate: Optional[str]
    failure_codes: List[str]
