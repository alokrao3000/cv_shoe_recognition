"""
Fetches a StockX product page's photo via a stealth browser session
(app/browser_session.py) — the official API returns no image field at all
(verified live: a catalog/search product has only productId/urlKey/styleId/
productType/title/brand/productAttributes), and a plain HTTP GET to the
product page is blocked by Cloudflare (403, a ~1MB challenge page, not the
real content). See app/browser_session.py's docstring for the full
rationale and real costs of the stealth-browser approach this depends on.
"""
import logging
import re
from typing import Optional

from app.browser_session import BrowserSession

logger = logging.getLogger(__name__)

# StockX serves product photos from this CDN regardless of exact page markup
# (which changes over time) — matching the domain directly on the rendered
# HTML is more robust than chasing a specific <img>/meta selector.
_IMAGE_URL_RE = re.compile(r'https://images\.stockx\.com/[^"\'\\\s)]+')


def fetch_product_image(session: BrowserSession, url_key: str) -> Optional[bytes]:
    """Navigates to stockx.com/{url_key} and returns the main product photo's
    bytes, or None on any failure (page didn't load, got a bot-check page
    instead of the real one, no image URL found, image download failed).
    Never raises for an ordinary miss — a backfill run should log and move
    on to the next candidate, not abort the whole batch over one bad page."""
    page = session.new_page()
    try:
        try:
            page.goto(f"https://stockx.com/{url_key}", wait_until="domcontentloaded")
            page.wait_for_timeout(1500)  # let client-rendered images finish attaching
        except Exception:
            logger.warning(f"[stockx-images] navigation failed for {url_key}")
            return None

        html = page.content()
        match = _IMAGE_URL_RE.search(html)
        if not match:
            logger.warning(f"[stockx-images] no images.stockx.com URL found on {url_key} "
                            f"(page shape may have changed, or the request was blocked)")
            return None
        image_url = match.group(0)

        try:
            resp = page.request.get(image_url)
        except Exception:
            logger.warning(f"[stockx-images] image download failed for {image_url}")
            return None
        if not resp.ok:
            logger.warning(f"[stockx-images] image fetch {image_url} returned {resp.status}")
            return None
        return resp.body()
    finally:
        page.close()
