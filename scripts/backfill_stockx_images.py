"""
Widens reference-catalog coverage beyond what sneaker-arbitrage's retailer
scrapes happen to include, by pulling real product photos straight from
StockX for SKUs the current index has no photo for.

Why this exists: the DB-sourced reference index (scripts/build_reference_
index.py) only ever contains SKUs some discount retailer scraped — a real
gap (see README §Known limitations; e.g. a legitimately StockX-sellable
Air Force 1 colorway that no tracked retailer happens to carry). StockX's
own catalog is far broader, but its official API returns no image field at
all, and its website is behind Cloudflare bot management — see
app/browser_session.py and app/stockx_images.py for how this gets a real
photo anyway, and the real costs (slow — a browser navigation per product,
not a plain HTTP request; still subject to network-reputation blocking with
no proxy configured) worth reading before running this at any real volume.

Flow:
  1. catalog/search a curated list of popular brand/silhouette terms (cheap,
     official API — counts against the normal StockX request budget).
  2. Skip anything already in the current reference index.
  3. For up to --max-new of what's left, fetch a real product photo via the
     stealth browser, embed it, and merge into the index. Paced with a
     delay between navigations (STOCKX_IMAGE_SCRAPE_DELAY_MIN/MAX) — this
     is the slow, rate-limited part, which is why --max-new defaults small.

Usage:
    python scripts/backfill_stockx_images.py [--max-new 50] [--headed]
"""
import argparse
import logging
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import embeddings, stockx_images                  # noqa: E402
from app.browser_session import BrowserSession               # noqa: E402
from app.config import settings                                # noqa: E402
from app.index import ReferenceIndex, ReferenceItem              # noqa: E402
from app.stockx_client import StockXAPIClient                     # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Deliberately not exhaustive — broad enough coverage of major current
# silhouettes/brands that catalog/search's relevance ranking surfaces a wide
# spread of colorways per term. Add to this list over time; there's no
# StockX "browse everything" endpoint to enumerate the catalog automatically.
SEED_QUERIES = [
    "Nike Air Force 1", "Nike Air Force 1 Low", "Nike Air Jordan 1",
    "Nike Air Jordan 3", "Nike Air Jordan 4", "Nike Air Jordan 11",
    "Nike Dunk Low", "Nike Dunk High", "Nike Air Max 1", "Nike Air Max 90",
    "Nike Air Max 95", "Nike Air Max 97", "Nike Cortez",
    "Adidas Yeezy 350", "Adidas Yeezy 500", "Adidas Samba", "Adidas Gazelle",
    "New Balance 550", "New Balance 990", "New Balance 2002R",
    "Asics Gel-Kayano 14", "Asics Gel-1130", "Converse Chuck 70",
    "Puma Suede", "Salomon XT-6", "On Cloudmonster", "Hoka Bondi",
]

_PAREN_RE = re.compile(r"\([^)]*\)")


def _first_style_segment(style_id: str) -> "str | None":
    """styleId can be compound ('315122-111/CW2288-111') — same convention
    as app/stockx_client.py's matching; store just the first, real segment."""
    for seg in (style_id or "").split("/"):
        seg = _PAREN_RE.sub(" ", seg).strip()
        if seg:
            return seg
    return None


def discover_candidates(client: StockXAPIClient, existing_skus: set) -> list[dict]:
    seen_norm = {seg.upper().replace("-", "").replace(" ", "") for seg in existing_skus}
    candidates = []
    for query in SEED_QUERIES:
        try:
            products = client._search(query, page_size=50)
        except Exception:
            logger.exception(f"catalog/search failed for {query!r} — skipping this term")
            continue
        for p in products:
            sku = _first_style_segment(p.get("styleId", ""))
            url_key = p.get("urlKey")
            if not sku or not url_key:
                continue
            norm = sku.upper().replace("-", "").replace(" ", "")
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            colorway = (p.get("productAttributes") or {}).get("colorway") or ""
            name = f"{p.get('title', '')} {colorway}".strip()
            candidates.append({"sku": sku, "url_key": url_key, "name": name})
        logger.info(f"'{query}': {len(products)} results, {len(candidates)} new candidates so far")
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=50,
                        help="Cap on how many new product photos to fetch this run "
                             "(each costs a real browser navigation — keep modest).")
    parser.add_argument("--headed", action="store_true",
                        help="Show the browser window (overrides BROWSER_HEADLESS) — "
                             "useful for a first run to sanity-check it isn't hitting a "
                             "Cloudflare challenge page.")
    args = parser.parse_args()

    existing = ReferenceIndex.load()
    logger.info(f"Current reference index: {len(existing)} SKUs.")

    stockx = StockXAPIClient()
    try:
        candidates = discover_candidates(stockx, existing.skus)
    finally:
        stockx.close()
    logger.info(f"Found {len(candidates)} candidate SKUs not already in the index.")
    if not candidates:
        logger.info("Nothing to backfill.")
        return

    random.shuffle(candidates)
    candidates = candidates[:args.max_new]
    logger.info(f"Fetching photos for {len(candidates)} of them (--max-new={args.max_new}) …")

    new_items, new_images = [], []
    fetched, failed = 0, 0
    with BrowserSession(platform="stockx", headless=not args.headed,
                         state_dir=settings.browser_state_dir,
                         nav_timeout_ms=settings.browser_nav_timeout_ms) as session:
        for i, c in enumerate(candidates):
            data = stockx_images.fetch_product_image(session, c["url_key"])
            if data is None:
                failed += 1
            else:
                try:
                    img = embeddings.load_image(data)
                    new_items.append(ReferenceItem(sku=c["sku"], name=c["name"],
                                                   image_url=f"https://stockx.com/{c['url_key']}"))
                    new_images.append(img)
                    fetched += 1
                except Exception:
                    logger.warning(f"Downloaded but couldn't decode image for {c['sku']}")
                    failed += 1

            if i < len(candidates) - 1:
                time.sleep(random.uniform(settings.stockx_image_scrape_delay_min,
                                          settings.stockx_image_scrape_delay_max))
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(candidates)} (fetched={fetched}, failed={failed})")

    logger.info(f"Done fetching: {fetched} photos, {failed} failed/skipped.")
    if not new_items:
        logger.info("No new images to add — index unchanged.")
        return

    logger.info("Embedding new images and merging into the reference index …")
    vectors = embeddings.embed_images(new_images)
    merged = existing.merged_with(new_items, vectors)
    merged.save()
    logger.info(f"Reference index grew from {len(existing)} to {len(merged)} SKUs.")


if __name__ == "__main__":
    main()
