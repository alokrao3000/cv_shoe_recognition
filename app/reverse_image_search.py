"""
Fallback SKU recovery for photos the local reference index can't confidently
match — see README's "Reference coverage gap" section. A shoe like a limited
collab (StrangeLove Dunks, say) is much more likely to show up on resale
sites than in ordinary retailer inventory, so the local index (built only
from sneaker-arbitrage's retailer scrapes) may simply never have seen it,
no matter how good the embedding match is.

Going straight at StockX for that photo is already blocked by Cloudflare
(see app/browser_session.py) — this instead goes through Google Images'
reverse-image-search upload flow to find OTHER pages carrying the same
photo, then scrapes a SKU/style-code off one of those pages. Any SKU
recovered this way still goes through the normal StockX catalog/search
match (app/stockx_client.py) before being trusted for a market lookup —
finding a SKU-shaped string on a random page is not itself proof it's
correct or that it's actually this shoe.

**Current status: blocked from THIS dev environment**, same class of
problem as the StockX image backfill. Verified live during development:
the very first reverse-image upload gets Google's own bot check
("Our systems have detected unusual traffic from your computer network",
landing on a google.com/sorry/index... reCAPTCHA page) before any results
ever render. This module does not attempt to solve or bypass that
challenge — search_by_image() detects it and reports a clear "blocked"
outcome instead of guessing past it. Two consequences:
  1. The result-page scraping below (best-guess text, result links, the
     labeled/bare SKU regexes) is UNVERIFIED against a real results page —
     written from Google Images' known layout, not confirmed live, and
     will likely need adjusting once it actually gets past the challenge.
  2. A residential proxy is the standard fix for this class of block (same
     as the StockX one) — thread one through REVERSE_IMAGE_SEARCH_PROXY_URL
     in .env, reusing BrowserSession's existing proxy support.
Run scripts/test_reverse_image_search.py --headed IMAGE_PATH on a network
where this doesn't trigger to see real results and fix up the selectors.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.browser_session import BrowserSession
from app.config import settings

logger = logging.getLogger(__name__)

# Domains a hit here is either the search engine itself or a site we already
# know is bot-walled the same way StockX is — not worth a navigation.
_SKIP_DOMAINS = ("google.com", "googleusercontent.com", "gstatic.com", "stockx.com", "goat.com")

# Conservative on purpose, same philosophy as app/index.py's classify_match:
# returning nothing is better than a wrong SKU silently feeding a price
# lookup. A labeled match ("Style: DD1391-100") is trusted outright; an
# unlabeled match is only trusted if it's the ONLY SKU-shaped string found —
# multiple candidates are too likely to include a false positive (a price, a
# date, an unrelated part number).
_LABELED_SKU_RE = re.compile(
    r"(?:style(?:\s*code|\s*id)?|sku|item\s*#|item\s*no\.?|product\s*code|mfg\s*#)\s*[:#]?\s*"
    r"([A-Z0-9]{5,9}-[A-Z0-9]{2,4})",
    re.IGNORECASE,
)
_BARE_SKU_RE = re.compile(r"\b[A-Z0-9]{5,9}-[A-Z0-9]{2,4}\b")


@dataclass
class ResultLink:
    url: str
    title: str = ""


@dataclass
class ReverseSearchOutcome:
    blocked: bool = False
    blocked_reason: Optional[str] = None
    best_guess: Optional[str] = None
    result_links: List[ResultLink] = field(default_factory=list)


def extract_sku_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    labeled = _LABELED_SKU_RE.search(text)
    if labeled:
        return labeled.group(1).upper()
    bare = {m.upper() for m in _BARE_SKU_RE.findall(text)}
    if len(bare) == 1:
        return bare.pop()
    return None


def _blocked_reason(page) -> Optional[str]:
    if "google.com/sorry" in page.url or "/sorry/index" in page.url:
        return "captcha_challenge"
    return None


def search_by_image(session: BrowserSession, image_bytes: bytes,
                     filename: str = "shoe.jpg") -> ReverseSearchOutcome:
    """Uploads image_bytes to Google Images' reverse-search and returns
    whatever it finds. Never raises for an ordinary miss/block — see
    ReverseSearchOutcome.blocked and this module's docstring."""
    page = session.new_page()
    try:
        try:
            page.goto("https://images.google.com/", wait_until="domcontentloaded",
                      timeout=session.nav_timeout_ms)
        except Exception:
            logger.warning("[reverse-image-search] navigation to images.google.com failed")
            return ReverseSearchOutcome(blocked=True, blocked_reason="navigation_failed")

        for text in ("Reject all", "I agree", "Accept all"):
            try:
                btn = page.get_by_role("button", name=text)
                if btn.count() > 0:
                    btn.first.click(timeout=3000)
                    page.wait_for_timeout(500)
                    break
            except Exception:
                pass

        file_input = page.locator("input[type=file]")
        if file_input.count() == 0:
            try:
                page.locator('[aria-label="Search by image"]').first.click(timeout=5000)
                page.wait_for_timeout(500)
            except Exception:
                logger.warning("[reverse-image-search] couldn't find the camera/upload control "
                                "— Google Images' page shape may have changed")
                return ReverseSearchOutcome(blocked=True, blocked_reason="upload_control_not_found")
            file_input = page.locator("input[type=file]")
        if file_input.count() == 0:
            return ReverseSearchOutcome(blocked=True, blocked_reason="upload_control_not_found")

        try:
            file_input.first.set_input_files(
                {"name": filename, "mimeType": "image/jpeg", "buffer": image_bytes},
                timeout=15000,
            )
            page.wait_for_timeout(3000)
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            logger.warning("[reverse-image-search] image upload/results wait failed")
            return ReverseSearchOutcome(blocked=True, blocked_reason="upload_failed")

        reason = _blocked_reason(page)
        if reason:
            logger.warning(f"[reverse-image-search] blocked: {reason} (url={page.url[:120]}...)")
            return ReverseSearchOutcome(blocked=True, blocked_reason=reason)

        # Best-effort scrape from here down — UNVERIFIED against a real
        # results page, see this module's docstring.
        best_guess = None
        for sel in ('div:has-text("Best guess for this image")', '[data-attrid="best-guess"]'):
            try:
                loc = page.locator(sel)
                if loc.count() > 0:
                    best_guess = loc.first.inner_text(timeout=2000).strip()
                    break
            except Exception:
                pass

        links: List[ResultLink] = []
        try:
            anchors = page.locator("a[href^='http']")
            n = min(anchors.count(), 40)
            for i in range(n):
                a = anchors.nth(i)
                href = a.get_attribute("href") or ""
                if not href or any(d in href for d in _SKIP_DOMAINS):
                    continue
                try:
                    title = (a.inner_text(timeout=1000) or "").strip()
                except Exception:
                    title = ""
                links.append(ResultLink(url=href, title=title))
                if len(links) >= settings.reverse_image_search_max_results:
                    break
        except Exception:
            logger.warning("[reverse-image-search] result-link scrape failed")

        return ReverseSearchOutcome(best_guess=best_guess, result_links=links)
    finally:
        page.close()


def find_sku(image_bytes: bytes, filename: str = "shoe.jpg") -> "tuple[Optional[str], Optional[str]]":
    """Orchestrates the whole fallback: reverse-image-search, then scan the
    best-guess text and top result pages for a SKU. Returns (sku, source) on
    success — source is 'best_guess' or the page URL the SKU came from.
    Returns (None, reason) otherwise, reason being 'blocked:<why>' or
    'no_sku_found'.

    Drives a real browser end-to-end (page loads, an image upload, possibly
    several more navigations) — this is seconds, not milliseconds. Call it
    off the asyncio event loop (e.g. `await asyncio.to_thread(find_sku, ...)`
    from the FastAPI endpoint), never directly from an async function."""
    with BrowserSession(platform="google_images", headless=settings.browser_headless,
                         proxy_url=settings.reverse_image_search_proxy_url,
                         state_dir=settings.browser_state_dir,
                         nav_timeout_ms=settings.browser_nav_timeout_ms) as session:
        outcome = search_by_image(session, image_bytes, filename)
        if outcome.blocked:
            return None, f"blocked:{outcome.blocked_reason}"

        if outcome.best_guess:
            sku = extract_sku_from_text(outcome.best_guess)
            if sku:
                return sku, "best_guess"

        for link in outcome.result_links:
            sku = extract_sku_from_text(link.title)
            if sku:
                return sku, link.url

        for link in outcome.result_links:
            page = session.new_page()
            try:
                page.goto(link.url, wait_until="domcontentloaded", timeout=session.nav_timeout_ms)
                page.wait_for_timeout(1000)
                text = page.inner_text("body", timeout=5000)
            except Exception:
                continue
            finally:
                page.close()
            sku = extract_sku_from_text(text)
            if sku:
                return sku, link.url

        return None, "no_sku_found"
