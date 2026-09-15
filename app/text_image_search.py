"""
Sources a real product photo for a known (sku, name) pair via a plain TEXT
image search — "Nike SB Dunk High Sweet Tooth Candy Corn FN5107-700" — not
a reverse-image upload. This is the sourcing half of the StockX-image-fetch
workaround: StockX's catalog/search (text) API is completely unblocked and
finds candidate SKUs fine (see scripts/backfill_stockx_images.py), it's
only fetching a PHOTO of each one *from StockX itself* that's walled by its
own bot management (see app/stockx_images.py's docstring). Since the shoe's
photo exists all over the web — retailer listings, eBay, sneaker blogs —
a text search for the SKU finds one without ever touching stockx.com.

Verified live: a plain Google Images text query for a real SKU+name (no
image upload involved) is NOT walled the way image-upload reverse search is
(see app/reverse_image_search.py's docstring for that contrast) — it
returns a normal results page with real product photos from real retailer/
resale pages. Much lower-risk than automating a reverse-image upload: this
is exactly what a human doing the same lookup manually would do.

Any image found this way is a real, if uncurated, photo of that SKU per
whatever page it was scraped from — there's no further verification that
it's the correct colorway beyond the search query itself being specific
(includes the SKU). Same trust model as the DB-sourced index (one retailer
photo per SKU, unverified) — see README's "Known limitations".
"""
import logging
import re
from typing import List, Optional
from urllib.parse import quote_plus

from patchright.sync_api import Page

from app.browser_session import BrowserSession
from app.config import settings

logger = logging.getLogger(__name__)


class SearchBlocked(Exception):
    """Raised when Google's own bot-check intercepts the search (a
    google.com/sorry/... page) — verified this happens after enough
    requests in one session/IP, not just on an image-upload flow (see this
    module's docstring). Distinct from an ordinary no-results miss so a
    caller can stop burning through remaining candidates once it starts,
    rather than paying the per-item delay for every one of them for no
    reason."""

# Google's own UI chrome/assets and low-res proxy thumbnails — never the
# actual product photo, always skip these regardless of what the page shows.
_SKIP_DOMAINS = (
    "gstatic.com", "google.com", "googleusercontent.com", "googleapis.com",
    "ggpht.com", "schema.org", "w3.org",
    # Already-walled domains — no point trying to fetch the image FROM here,
    # even if a search result happens to link to one.
    "stockx.com", "goat.com",
)

_IMG_URL_RE = re.compile(r'"(https://[^"]+?\.(?:jpg|jpeg|png|webp))"', re.IGNORECASE)


def _candidate_urls(html: str, max_results: int) -> List[str]:
    seen, out = set(), []
    for url in _IMG_URL_RE.findall(html):
        if url in seen or any(d in url for d in _SKIP_DOMAINS):
            continue
        seen.add(url)
        out.append(url)
        if len(out) >= max_results:
            break
    return out


def search_image_urls(page: Page, query: str, max_results: int = 5) -> List[str]:
    """Returns up to max_results candidate photo URLs for query, best (most
    relevant per Google's own ranking) first. Empty list on an ordinary
    miss/navigation failure. Raises SearchBlocked (see that class) if
    Google's bot-check intercepted the request instead of running it —
    that's not a per-query miss, it means every subsequent call in this
    session will fail the same way until it clears."""
    try:
        page.goto(f"https://www.google.com/search?q={quote_plus(query)}&udm=2",
                   wait_until="domcontentloaded", timeout=20000)
        page.wait_for_timeout(1500)
    except Exception:
        logger.warning(f"[text-image-search] navigation failed for {query!r}")
        return []
    if "google.com/sorry" in page.url:
        raise SearchBlocked(f"Google's bot-check intercepted the search (url={page.url[:120]}...)")
    return _candidate_urls(page.content(), max_results)


def fetch_first_working_image(session: BrowserSession, query: str,
                              max_candidates: int = 5) -> Optional[bytes]:
    """Searches for query, then tries each candidate URL in ranked order
    until one actually downloads — a listed URL can 404/expire even though
    the search result itself was real. Returns the image bytes, or None if
    nothing usable was found."""
    page = session.new_page()
    try:
        urls = search_image_urls(page, query, max_candidates)
        for url in urls:
            try:
                resp = page.request.get(url, timeout=15000)
            except Exception:
                continue
            if resp.ok:
                return resp.body()
        return None
    finally:
        page.close()
