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
    # "cls" | "mean_patch" | "cls_mean_concat" — empirically compared with
    # scripts/eval_pooling.py on a real cross-retailer-photo leave-out task.
    # CLS won clearly (57.5% top-1 vs 22.5% for mean_patch, 51.2% for the
    # concat) — patch-token pooling turned out to carry more background/
    # crop/angle noise than useful fine-grained signal on real retailer
    # photos. Don't switch this without re-running that eval on real data.
    embedding_pooling: str = "cls"
    reference_index_dir: str = str(BASE_DIR / "data" / "reference_index")
    query_cache_dir: str = str(BASE_DIR / "data" / "query_cache")

    # A predicted SKU is only trusted when the top candidate clears this
    # cosine-similarity floor AND beats the runner-up by min_match_margin —
    # see app/index.py::search(). A wrong SKU silently feeding a price lookup
    # is worse than returning "unidentified".
    min_match_similarity: float = 0.80
    min_match_margin: float = 0.03
    top_k: int = 5

    # ── Coverage backfill (scripts/backfill_stockx_images.py) ──
    # The reference catalog built from sneaker-arbitrage's DB only covers
    # SKUs some retailer happened to scrape (see README §Known limitations).
    # This fills gaps by fetching real StockX product photos through a
    # stealth browser (app/browser_session.py) — see that file's docstring
    # for why a plain HTTP fetch doesn't work and what it actually costs.
    browser_headless: bool = True
    browser_state_dir: str = str(BASE_DIR / "data" / "browser_state")
    browser_nav_timeout_ms: int = 30000
    # Delay between product-page navigations — pacing, not speed; StockX's
    # bot-management weighs request cadence as well as browser fingerprint.
    stockx_image_scrape_delay_min: float = 2.0
    stockx_image_scrape_delay_max: float = 5.0

    # ── Reverse-image-search SKU fallback (app/reverse_image_search.py) ──
    # When the local reference index has no confident match, optionally try
    # recovering a SKU via Google Images reverse search instead of just
    # returning identified=False. Off by default: verified BLOCKED by
    # Google's own bot check from this dev environment (see that module's
    # docstring) — enable only once you've confirmed it actually gets past
    # that on your network (scripts/test_reverse_image_search.py --headed).
    reverse_image_search_enabled: bool = False
    reverse_image_search_max_results: int = 5
    reverse_image_search_proxy_url: str = ""


settings = Settings()
