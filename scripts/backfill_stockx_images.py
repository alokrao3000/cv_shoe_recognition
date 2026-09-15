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

# Deliberately not exhaustive — there's no StockX "browse everything"
# endpoint to enumerate the catalog automatically, so this is broad
# silhouette/brand coverage for catalog/search's relevance ranking to surface
# a wide spread of colorways from, not a full catalog. Combined with
# PAGES_PER_QUERY below (StockX caps pageSize at 50, verified live) this is
# the actual ceiling on what one run can ever discover — add more terms
# (specific collabs included) over time to raise it further.
SEED_QUERIES = [
    # Nike / Jordan
    "Nike Air Force 1", "Nike Air Force 1 Low", "Nike Air Force 1 High",
    "Nike Air Force 1 Shadow", "Air Jordan 1", "Air Jordan 1 Low",
    "Air Jordan 1 High", "Air Jordan 2", "Air Jordan 3", "Air Jordan 4",
    "Air Jordan 5", "Air Jordan 6", "Air Jordan 7", "Air Jordan 9",
    "Air Jordan 11", "Air Jordan 12", "Air Jordan 13", "Air Jordan 14",
    "Travis Scott Jordan", "Off-White Jordan", "Union Jordan",
    "Fragment Jordan", "Nike Dunk Low", "Nike Dunk High", "Nike SB Dunk Low",
    "Nike SB Dunk High", "StrangeLove Dunk", "Off-White Dunk", "Chunky Dunky",
    "Travis Scott Dunk", "Parra Dunk", "Nike Air Max 1", "Nike Air Max 90",
    "Nike Air Max 95", "Nike Air Max 97", "Nike Air Max 98", "Nike Air Max 270",
    "Nike Air Max Plus", "Nike Air Max 720", "Nike Cortez", "Nike Blazer Mid",
    "Nike Blazer Low", "Nike Vapormax", "Nike P-6000", "Nike Shox",
    # Adidas
    "Adidas Yeezy 350", "Adidas Yeezy 500", "Adidas Yeezy 700",
    "Adidas Yeezy 380", "Adidas Yeezy 450", "Adidas Yeezy Slide",
    "Adidas Yeezy Foam Runner", "Adidas Samba", "Adidas Gazelle",
    "Adidas Superstar", "Adidas Stan Smith", "Adidas Campus",
    "Adidas Ultraboost", "Adidas NMD", "Adidas Forum Low", "Adidas Forum High",
    "Adidas Handball Spezial",
    # New Balance
    "New Balance 550", "New Balance 574", "New Balance 990",
    "New Balance 991", "New Balance 992", "New Balance 993",
    "New Balance 996", "New Balance 997", "New Balance 2002R",
    "New Balance 327", "New Balance 9060", "New Balance 1906R",
    # Asics
    "Asics Gel-Kayano 14", "Asics Gel-1130", "Asics Gel-NYC",
    "Asics Gel-Lyte III", "Asics Gel-1090",
    # Other brands
    "Converse Chuck 70", "Converse Chuck Taylor", "Vans Old Skool",
    "Vans Sk8-Hi", "Vans Authentic", "Puma Suede", "Puma RS-X",
    "Puma Palermo", "Puma Speedcat", "Reebok Classic Leather",
    "Reebok Club C", "Reebok Question", "Salomon XT-6", "Salomon XT-4",
    "On Cloudmonster", "On Cloud 5", "Hoka Bondi", "Hoka Clifton",
    "Under Armour Curry", "Crocs Classic Clog", "Timberland 6 Inch Boot",
]

# Each term above only returns up to pageSize (50, StockX's own cap) results
# per catalog/search call by default; this pulls PAGES_PER_QUERY pages per
# term (paced the same as everything else here) to get further into each
# term's relevance ranking instead of only ever seeing the top 50.
PAGES_PER_QUERY = 3

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
        query_new = 0
        for page in range(1, PAGES_PER_QUERY + 1):
            try:
                products = client._search(query, page_size=50, page_number=page)
            except Exception:
                logger.exception(f"catalog/search failed for {query!r} page {page} — moving to next term")
                break
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
                query_new += 1
            if len(products) < 50:
                break  # fewer than a full page back — this term is exhausted
        logger.info(f"'{query}': {query_new} new candidates so far, {len(candidates)} total")
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
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="Embed + save the index to disk every N successful fetches, "
                             "instead of only once at the end — a long low-yield run "
                             "(StockX's bot-wall makes most attempts fail, see README) "
                             "shouldn't lose everything to an interruption.")
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

    def flush(pending_items, pending_images):
        """Embeds + merges + saves whatever's pending so far — called
        periodically, not just at the end, so a long low-yield run doesn't
        lose everything to an interruption. Returns the new running index."""
        nonlocal existing
        if not pending_items:
            return
        logger.info(f"Checkpoint: embedding {len(pending_items)} pending image(s) and saving …")
        vectors = embeddings.embed_images(pending_images)
        existing = existing.merged_with(pending_items, vectors)
        existing.save()
        logger.info(f"Reference index now at {len(existing)} SKUs (checkpoint saved).")

    pending_items, pending_images = [], []
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
                    pending_items.append(ReferenceItem(sku=c["sku"], name=c["name"],
                                                       image_url=f"https://stockx.com/{c['url_key']}"))
                    pending_images.append(img)
                    fetched += 1
                except Exception:
                    logger.warning(f"Downloaded but couldn't decode image for {c['sku']}")
                    failed += 1

            if len(pending_items) >= args.checkpoint_every:
                flush(pending_items, pending_images)
                pending_items, pending_images = [], []

            if i < len(candidates) - 1:
                time.sleep(random.uniform(settings.stockx_image_scrape_delay_min,
                                          settings.stockx_image_scrape_delay_max))
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(candidates)} (fetched={fetched}, failed={failed}, "
                            f"index now {len(existing) + len(pending_items)} SKUs)")

        flush(pending_items, pending_images)

    logger.info(f"Done fetching: {fetched} photos, {failed} failed/skipped. "
                f"Reference index: {len(existing)} SKUs.")


if __name__ == "__main__":
    main()
