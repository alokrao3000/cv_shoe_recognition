from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── This project's own database (Postgres + pgvector, docker-compose.yml) ──
    database_url: str = "postgresql+psycopg2://cvshoe:cvshoe_dev@localhost:5434/cv_shoe_recognition"

    # Read-only source for seeding reference products/images (sneaker-arbitrage
    # DB) — only scripts/seed_from_arbitrage_db.py and build_eval_set.py use it.
    arbitrage_database_url: str = "postgresql://postgres@localhost:5433/sneaker_arbitrage"

    host: str = "0.0.0.0"
    port: int = 8100

    # ── API security ──
    # Optional bearer token required on /api/* when set. Leave empty for local use.
    api_token: str = ""
    rate_limit_per_minute: int = 60
    max_upload_bytes: int = 15 * 1024 * 1024
    max_images_per_request: int = 6
    fetch_timeout_seconds: float = 20.0
    data_dir: str = str(BASE_DIR / "data")

    # ── StockX official API (developer.stockx.com) ──
    stockx_client_id: str = ""
    stockx_client_secret: str = ""
    stockx_api_key: str = ""
    stockx_refresh_token: str = ""
    stockx_redirect_uri: str = "http://localhost:8018/stockx/callback"
    stockx_token_cache_path: str = str(BASE_DIR / "data" / "stockx_token.json")
    stockx_cache_ttl_hours: int = 24

    # ── Vision LLM (Anthropic) ──
    # ANTHROPIC_API_KEY is read by the SDK itself from the environment; this
    # setting exists so a .env file works too.
    anthropic_api_key: str = ""
    vision_enabled: bool = True
    vision_model: str = "claude-opus-5"
    vision_max_image_px: int = 1568

    # ── Web search provider ──
    search_provider: str = "serper"        # serper | none
    serper_api_key: str = ""
    web_search_max_results: int = 10

    # ── Embeddings ──
    embedding_model: str = "facebook/dinov2-base"
    embedding_pooling: str = "cls"          # see app/vision/embeddings.py — don't change without re-running the eval
    embedding_dim: int = 768
    embedding_top_k: int = 10
    # Cosine similarity below this is treated as "no visual evidence" rather than
    # weak evidence — DINOv2 similarities between unrelated shoes cluster ~0.5-0.7.
    embedding_floor: float = 0.70
    # ...and at/above this it counts as full visual agreement (the real
    # cross-retailer eval put correct matches around 0.85-0.95).
    embedding_ceiling: float = 0.92

    # ── Identity-match scoring weights (sum to 1.0; see app/pipeline/scoring.py) ──
    weight_sku_match: float = 0.40
    weight_model_match: float = 0.20
    weight_colorway_match: float = 0.15
    weight_embedding_similarity: float = 0.15
    weight_metadata_match: float = 0.05
    weight_external_agreement: float = 0.05

    # ── Confidence tiers ──
    confidence_high: float = 0.95
    confidence_medium: float = 0.85
    confidence_low: float = 0.70
    # Top-1 must beat top-2 by this much, else MULTIPLE_CANDIDATES.
    ambiguity_margin: float = 0.05
    verification_top_n: int = 3

    # ── Resolution cache ──
    resolution_cache_ttl_hours: int = 24 * 7


settings = Settings()
