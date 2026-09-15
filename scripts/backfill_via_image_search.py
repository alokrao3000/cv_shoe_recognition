"""
Widens reference-catalog coverage the same way scripts/backfill_stockx_
images.py does (SKUs the sneaker-arbitrage-sourced index never saw a photo
for — see README §Reference coverage gap), but sources each photo via a
plain TEXT image search (app/text_image_search.py) instead of fetching it
from stockx.com directly.

Why this exists as a separate script rather than a flag on the other one:
StockX's own site is walled by its bot management for the image-fetch step
(verified live — see app/stockx_images.py's docstring and README), to the
point of blocking 100% of attempts in one real run. Candidate DISCOVERY via
StockX's catalog/search API is completely unaffected (it's a normal,
official, rate-limited API call) — this script reuses that part unchanged
(discover_candidates(), same SEED_QUERIES) and only replaces the walled
image-fetch step with a text search for "{name} {sku}" that finds the same
photo on some other, unwalled page (eBay, a retailer, a sneaker blog).

Usage:
    python scripts/backfill_via_image_search.py [--max-new 500] [--headed]
"""
import argparse
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import embeddings, text_image_search                       # noqa: E402
from app.text_image_search import SearchBlocked                      # noqa: E402
from app.browser_session import BrowserSession                       # noqa: E402
from app.config import settings                                       # noqa: E402
from app.index import ReferenceIndex, ReferenceItem                    # noqa: E402
from app.stockx_client import StockXAPIClient                           # noqa: E402
from scripts.backfill_stockx_images import discover_candidates           # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=500,
                        help="Cap on how many candidate SKUs to attempt this run.")
    parser.add_argument("--headed", action="store_true",
                        help="Show the browser window — useful for a first run to "
                             "sanity-check the results look right.")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="Embed + save the index to disk every N successful fetches, "
                             "instead of only once at the end.")
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
    logger.info(f"Sourcing photos for {len(candidates)} of them via text image search "
                f"(--max-new={args.max_new}) …")

    def flush(pending_items, pending_images):
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
    with BrowserSession(platform="text_image_search", headless=not args.headed,
                         state_dir=settings.browser_state_dir,
                         nav_timeout_ms=settings.browser_nav_timeout_ms) as session:
        for i, c in enumerate(candidates):
            query = f"{c['name']} {c['sku']}".strip()
            try:
                data = text_image_search.fetch_first_working_image(session, query)
            except SearchBlocked as exc:
                logger.warning(f"Stopping early: {exc}. Google's bot-check kicked in after "
                                f"{fetched} successful fetches this session — grinding through "
                                f"the remaining {len(candidates) - i} candidates would just fail "
                                f"the same way. Saving what's been found so far.")
                break
            if data is None:
                failed += 1
            else:
                try:
                    img = embeddings.load_image(data)
                    pending_items.append(ReferenceItem(sku=c["sku"], name=c["name"],
                                                       image_url=f"text-image-search:{query}"))
                    pending_images.append(img)
                    fetched += 1
                except Exception:
                    logger.warning(f"Downloaded but couldn't decode image for {c['sku']}")
                    failed += 1

            if len(pending_items) >= args.checkpoint_every:
                flush(pending_items, pending_images)
                pending_items, pending_images = [], []

            if i < len(candidates) - 1:
                time.sleep(random.uniform(settings.image_search_delay_min,
                                          settings.image_search_delay_max))
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(candidates)} (fetched={fetched}, failed={failed}, "
                            f"index now {len(existing) + len(pending_items)} SKUs)")

        flush(pending_items, pending_images)

    logger.info(f"Done: {fetched} photos found, {failed} failed/skipped. "
                f"Reference index: {len(existing)} SKUs.")


if __name__ == "__main__":
    main()
