"""
Seeds products + reference images from the sneaker-arbitrage Postgres DB:
every (sku, image_url) its scrapers captured is a free, real-world labeled
photo. Multiple retailer photos per SKU are kept (up to --max-per-sku) —
more angles/lighting per product makes the visual signal more robust.

Usage:
    python scripts/seed_from_arbitrage_db.py [--limit N] [--max-per-sku 3] [--workers 8]

Requires both databases: this project's (docker compose up -d postgres) and
sneaker-arbitrage's at ARBITRAGE_DATABASE_URL.
"""
import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import psycopg2
import psycopg2.extras

from app.config import settings                                   # noqa: E402
from app.db import repo                                            # noqa: E402
from app.db.session import db_session, init_db                     # noqa: E402
from app.extraction.sku_detector import candidate_from_identifier  # noqa: E402
from app.vision import embeddings                                  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def fetch_rows(limit, max_per_sku):
    conn = psycopg2.connect(settings.arbitrage_database_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT sku, name, image_url, product_url, original_price, supplier_id, scraped_at
                FROM supplier_products
                WHERE sku IS NOT NULL AND sku != '' AND image_url IS NOT NULL AND image_url != ''
                ORDER BY sku, scraped_at DESC
            """)
            rows = cur.fetchall()
    finally:
        conn.close()
    per_sku, out, seen = {}, [], set()
    for r in rows:
        key = (r["sku"], r["image_url"].split("?")[0])
        if key in seen:
            continue
        seen.add(key)
        n = per_sku.get(r["sku"], 0)
        if n >= max_per_sku:
            continue
        per_sku[r["sku"]] = n + 1
        out.append(dict(r))
    if limit:
        keep = set(list(dict.fromkeys(r["sku"] for r in out))[:limit])
        out = [r for r in out if r["sku"] in keep]
    return out


def download(url, http):
    try:
        resp = http.get(url, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Only the first N distinct SKUs.")
    ap.add_argument("--max-per-sku", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64, help="Images embedded + committed per batch.")
    ap.add_argument("--exclude-eval", action="store_true",
                    help="Skip image URLs used as queries in data/eval/cases.jsonl (no leakage into the index).")
    ap.add_argument("--only-eval-skus", action="store_true",
                    help="Only seed SKUs that appear in data/eval/cases.jsonl (their non-query photos).")
    args = ap.parse_args()

    init_db()
    rows = fetch_rows(args.limit, args.max_per_sku)
    if args.exclude_eval or args.only_eval_skus:
        import json
        from app.extraction.sku_detector import normalize_style_code
        cases_path = Path(settings.data_dir) / "eval" / "cases.jsonl"
        held_out, eval_skus = set(), set()
        if cases_path.exists():
            for line in cases_path.read_text(encoding="utf-8").splitlines():
                if line.strip() and not line.startswith("#"):
                    case = json.loads(line)
                    held_out.add((case.get("image_url") or "").split("?")[0])
                    eval_skus.add(normalize_style_code(case.get("expected_style_code", "")))
        before = len(rows)
        rows = [r for r in rows if r["image_url"].split("?")[0] not in held_out]
        logger.info(f"--exclude-eval: dropped {before - len(rows)} eval query images")
        if args.only_eval_skus:
            rows = [r for r in rows if normalize_style_code(r["sku"]) in eval_skus]
            logger.info(f"--only-eval-skus: {len(rows)} rows for {len(eval_skus)} eval SKUs")
    logger.info(f"{len(rows)} (sku, image) rows from the arbitrage DB")

    with db_session() as s:
        existing = {(img.product_id, img.image_url.split("?")[0]) for img in
                    s.execute(__import__("sqlalchemy").select(repo.ProductImage)).scalars()}
    logger.info(f"{len(existing)} reference images already stored")

    added, skipped, failed = 0, 0, 0
    with httpx.Client() as http, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(rows), args.batch):
            batch = rows[start:start + args.batch]
            futures = {pool.submit(download, r["image_url"], http): r for r in batch}
            items = []
            for fut in as_completed(futures):
                r = futures[fut]
                data = fut.result()
                if not data:
                    failed += 1
                    continue
                try:
                    img = embeddings.load_image(data)
                except Exception:
                    failed += 1
                    continue
                items.append((r, data, img))
            if not items:
                continue
            vecs = embeddings.embed_images([i[2] for i in items])
            with db_session() as s:
                for (r, data, _), vec in zip(items, vecs):
                    cand = candidate_from_identifier(r["sku"], "sku", "arbitrage")
                    code = cand.code if cand else r["sku"].upper().replace("-", "").replace(" ", "")
                    display = cand.display if cand else r["sku"]
                    product = repo.upsert_product(s, code, name=r.get("name") or "", source="retailer", display=display,
                                                  retail_price=float(r["original_price"]) if r.get("original_price") else None)
                    if (product.id, r["image_url"].split("?")[0]) in existing:
                        skipped += 1
                        continue
                    repo.add_product_image(s, product, vec, image_url=r["image_url"], image_bytes=data, source="retailer")
                    existing.add((product.id, r["image_url"].split("?")[0]))
                    added += 1
            logger.info(f"{min(start + args.batch, len(rows))}/{len(rows)} — added {added}, skipped {skipped}, failed {failed}")
    logger.info(f"Done: {added} images added, {skipped} already present, {failed} failed.")


if __name__ == "__main__":
    main()
