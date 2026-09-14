"""
Stealth-browser session for fetching StockX product pages — ported and
trimmed from sneaker-arbitrage's app/scrapers/browser.py. See that file's
docstring for the full rationale; not repeated here except where this port
differs.

Why this exists at all: StockX's official API (app/stockx_client.py) has no
image field — confirmed by hitting it live (see scripts/backfill_stockx_
images.py's docstring). Getting real photos means the actual website, which
is behind Cloudflare bot management — a plain httpx GET gets a 403 challenge
page. patchright (a patched Chromium build, not stock Playwright) defeats
the CDP-level fingerprinting that trips it; this was already proven out in
sneaker-arbitrage for both StockX and GOAT market-data scraping.

Trimmed vs. the original:
  - No ThreadBoundProxy / uvicorn-asyncio thread-affinity workaround — that
    problem is specific to running Playwright inside the same process as an
    asyncio-based server. This project's browser use is confined to a
    one-shot CLI backfill script (scripts/backfill_stockx_images.py), never
    the FastAPI app, so there's no conflicting event loop to isolate it from.
  - No capture_json_response helper — StockX product pages are scraped for
    their rendered HTML/image URL, not an intercepted API response.
  - Own browser_state dir (data/browser_state), separate from sneaker-
    arbitrage's — two processes writing the same persisted-cookie file could
    race. Starting cold each time this project's session is reused across
    runs is an acceptable cost for an occasional backfill job.

Real, non-hypothetical costs of using this, worth reading before running the
backfill at any real volume: it drives an actual browser per product page
(slow — seconds per page, not the milliseconds a plain HTTP request would
take), and it's still subject to network-reputation blocking (Cloudflare
weighs source IP, not just browser fingerprint) — expect materially worse
reliability run headless on a residential IP with no proxy than one might
assume from the sneaker-arbitrage README's own caveat on this. Pace requests
(see STOCKX_IMAGE_SCRAPE_DELAY_MIN/MAX in .env) and keep backfill batches
modest.
"""
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlsplit

from patchright.sync_api import sync_playwright, BrowserContext, Page

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _proxy_config(proxy_url: str) -> dict:
    parsed = urlsplit(proxy_url)
    netloc = parsed.hostname or ""
    if parsed.port:
        netloc += f":{parsed.port}"
    config = {"server": f"{parsed.scheme}://{netloc}"}
    if parsed.username:
        config["username"] = unquote(parsed.username)
    if parsed.password:
        config["password"] = unquote(parsed.password)
    return config


class BrowserSession:
    """One persistent Chromium context. Use as a context manager around a
    batch of page fetches — reopening a full browser per page would be much
    slower and would defeat storage-state (cookie) reuse mid-run."""

    def __init__(self, platform: str = "stockx", headless: bool = True, proxy_url: str = "",
                 state_dir: str = "data/browser_state", nav_timeout_ms: int = 30000):
        self.platform = platform
        self.headless = headless
        self.proxy_url = proxy_url
        self.nav_timeout_ms = nav_timeout_ms
        self._state_path = Path(state_dir) / f"{platform}.json"
        self._playwright = None
        self._browser = None
        self._context: Optional[BrowserContext] = None

    def start(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context_kwargs = {
            "user_agent": _USER_AGENT,
            "viewport": {"width": 1920, "height": 1080},
            "locale": "en-US",
            "timezone_id": "America/New_York",
        }
        if self._state_path.exists():
            context_kwargs["storage_state"] = str(self._state_path)
        if self.proxy_url:
            context_kwargs["proxy"] = _proxy_config(self.proxy_url)
        self._context = self._browser.new_context(**context_kwargs)
        self._context.set_default_navigation_timeout(self.nav_timeout_ms)
        logger.info(f"[{self.platform}] browser session started "
                    f"(headless={self.headless}, resumed_state={self._state_path.exists()})")

    def new_page(self) -> Page:
        return self._context.new_page()

    def save_state(self) -> None:
        if self._context is None:
            return
        try:
            self._context.storage_state(path=str(self._state_path))
        except Exception:
            logger.exception(f"[{self.platform}] failed to persist browser storage state")

    def close(self) -> None:
        self.save_state()
        try:
            if self._context:
                self._context.close()
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception:
            logger.exception(f"[{self.platform}] error tearing down browser session")

    def __enter__(self) -> "BrowserSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
