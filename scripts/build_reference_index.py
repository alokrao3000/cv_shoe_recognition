"""
Builds the reference embedding index from the sneaker-arbitrage Postgres DB:
for every distinct SKU that DB's scrapers successfully parsed (supplier_
products.sku is populated), pull its most recent retailer photo, embed it,
and write a FAISS index + metadata to data/reference_index/.

This is the ONE place this project talks to that DB, and it's read-only.
Run it once to bootstrap, then periodically (e.g. weekly, or after a big
scrape run) to pick up newly-seen SKUs — it's a full rebuild, not
incremental, which is fine at this scale (thousands of SKUs, a few minutes).

Usage:
    python scripts/build_reference_index.py [--limit N] [--workers 8]

Requires the sneaker-arbitrage Postgres container to be reachable at
ARBITRAGE_DATABASE_URL (.env) — start it from that repo with
`docker compose up -d postgres` (or the `runprogram` command there) if
this script can't connect.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import psycopg2
import psycopg2.extras
from concurrent.futures import ThreadPoolExecutor, as_completed

from app import embeddings                                # noqa: E402
from app.config import settings                            # noqa: E402
from app.index import ReferenceIndex, ReferenceItem         # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IMAGE_TIMEOUT = 15.0
EMBED_BATCH_SIZE = 16


def fetch_candidates(limit: "int | None") -> list[dict]:
    """One row per distinct SKU — most recently scraped image_url wins.
    DISTINCT ON requires the ORDER BY to lead with the same column."""
    query = """
        SELECT DISTINCT ON (sku) sku, name, image_url
        FROM supplier_products
        WHERE sku IS NOT NULL AND sku != '' AND image_url IS NOT NULL AND image_url != ''
        ORDER BY sku, scraped_at DESC
    """
    if limit:
        query = f"SELECT * FROM ({query}) sub LIMIT {int(limit)}"

    conn = psycopg2.connect(settings.arbitrage_database_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def download_image(url: str, http: httpx.Client) -> "bytes | None":
    try:
        resp = http.get(url, timeout=IMAGE_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        logger.debug(f"Download failed for {url}: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N distinct SKUs (for a quick test build).")
    parser.add_argument("--workers", type=int, default=8, help="Parallel image downloads.")
    args = parser.parse_args()

    logger.info("Fetching (sku, name, image_url) candidates from the sneaker-arbitrage DB …")
    rows = fetch_candidates(args.limit)
    logger.info(f"Got {len(rows)} distinct SKUs with an image.")
    if not rows:
        sys.exit("No candidate rows found — is the sneaker-arbitrage DB populated and reachable?")

    items: list[ReferenceItem] = []
    images = []
    failed = 0

    with httpx.Client() as http, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_image, row["image_url"], http): row for row in rows}
        done = 0
        for future in as_completed(futures):
            row = futures[future]
            done += 1
            if done % 200 == 0:
                logger.info(f"Downloaded {done}/{len(rows)} …")
            data = future.result()
            if data is None:
                failed += 1
                continue
            try:
                img = embeddings.load_image(data)
            except Exception:
                failed += 1
                continue
            images.append(img)
            items.append(ReferenceItem(sku=row["sku"], name=row.get("name"), image_url=row["image_url"]))

    logger.info(f"Downloaded {len(images)} images ({failed} failed/skipped). Embedding …")
    if not images:
        sys.exit("No images downloaded successfully — nothing to index.")

    vectors = embeddings.embed_images(images, batch_size=EMBED_BATCH_SIZE)
    index = ReferenceIndex.build(items, vectors)
    index.save()
    logger.info(f"Done — reference index has {len(index)} SKUs, saved to {settings.reference_index_dir}")


if __name__ == "__main__":
    main()
