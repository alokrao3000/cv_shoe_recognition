from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Read-only source for reference images (the sneaker-arbitrage DB) —
    # only used by scripts/build_reference_index.py.
    arbitrage_database_url: str = "postgresql://arbitrage:arbitrage_dev@localhost:5432/sneaker_arbitrage"

    host: str = "0.0.0.0"
    port: int = 8100

    # ── StockX official API (developer.stockx.com) ──
    # Same shape as sneaker-arbitrage's app/config.py, but this project keeps
    # its own refresh token (see .env.example) rather than sharing one, and
    # persists rotations to a local JSON file instead of a DB table — see
    # app/stockx_client.py.
    stockx_client_id: str = ""
    stockx_client_secret: str = ""
    stockx_api_key: str = ""
    stockx_refresh_token: str = ""
    stockx_redirect_uri: str = "http://localhost:8018/stockx/callback"
    stockx_token_cache_path: str = str(BASE_DIR / "data" / "stockx_token.json")

    # ── Identification ──
    embedding_model: str = "facebook/dinov2-base"
    reference_index_dir: str = str(BASE_DIR / "data" / "reference_index")
    query_cache_dir: str = str(BASE_DIR / "data" / "query_cache")

    # A predicted SKU is only trusted when the top candidate clears this
    # cosine-similarity floor AND beats the runner-up by min_match_margin —
    # see app/index.py::search(). A wrong SKU silently feeding a price lookup
    # is worse than returning "unidentified".
    min_match_similarity: float = 0.80
    min_match_margin: float = 0.03
    top_k: int = 5


settings = Settings()
