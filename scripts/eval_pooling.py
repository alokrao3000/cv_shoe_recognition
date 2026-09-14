"""
Empirically compares embedding-pooling strategies (app/embeddings.POOLINGS)
on a real leave-out task, instead of guessing which one is more discriminative.

Method: pick SKUs that have 2+ distinct photos from DIFFERENT retailers in
the sneaker-arbitrage DB. For each, one photo goes in the reference set, a
DIFFERENT photo of the SAME shoe is held out as the query — so this measures
whether the embedding recognizes the same shoe across a genuinely different
photo (different retailer, lighting, crop), which is exactly what real
identify requests need. A pile of other single-image SKUs is added as
distractors so the retrieval task is realistic (matching against thousands
of candidates, not just the 1:1 pairs).

Usage:
    python scripts/eval_pooling.py [--eval-skus 80] [--distractors 400]
"""
import argparse
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import numpy as np
import psycopg2
import psycopg2.extras
from concurrent.futures import ThreadPoolExecutor, as_completed

from app import embeddings                          # noqa: E402
from app.config import settings                       # noqa: E402
from app.embeddings import POOLINGS                    # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def fetch_pairs(n: int) -> list[dict]:
    """n SKUs with 2+ distinct images; returns {sku, ref_url, query_url}."""
    conn = psycopg2.connect(settings.arbitrage_database_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT sku, array_agg(DISTINCT image_url) AS urls
                FROM supplier_products
                WHERE sku IS NOT NULL AND sku != '' AND image_url IS NOT NULL AND image_url != ''
                GROUP BY sku HAVING COUNT(DISTINCT image_url) >= 2
            """)
            rows = cur.fetchall()
    finally:
        conn.close()
    random.Random(RANDOM_SEED).shuffle(rows)
    out = []
    for row in rows[:n]:
        urls = row["urls"]
        ref_url, query_url = urls[0], urls[1]
        out.append({"sku": row["sku"], "ref_url": ref_url, "query_url": query_url})
    return out


def fetch_distractors(n: int, exclude_skus: set) -> list[dict]:
    conn = psycopg2.connect(settings.arbitrage_database_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT DISTINCT ON (sku) sku, image_url
                FROM supplier_products
                WHERE sku IS NOT NULL AND sku != '' AND image_url IS NOT NULL AND image_url != ''
                ORDER BY sku, scraped_at DESC
            """)
            rows = [dict(r) for r in cur.fetchall() if r["sku"] not in exclude_skus]
    finally:
        conn.close()
    random.Random(RANDOM_SEED + 1).shuffle(rows)
    return rows[:n]


def download_all(urls: list[str], workers: int = 8) -> dict:
    """url -> image bytes (or None on failure)."""
    results = {}
    with httpx.Client() as http, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(http.get, u, timeout=15.0, follow_redirects=True): u for u in urls}
        for future in as_completed(futures):
            u = futures[future]
            try:
                resp = future.result()
                resp.raise_for_status()
                results[u] = resp.content
            except Exception:
                results[u] = None
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-skus", type=int, default=80)
    parser.add_argument("--distractors", type=int, default=400)
    args = parser.parse_args()

    logger.info(f"Fetching {args.eval_skus} eval pairs + {args.distractors} distractors …")
    pairs = fetch_pairs(args.eval_skus)
    distractors = fetch_distractors(args.distractors, exclude_skus={p["sku"] for p in pairs})
    logger.info(f"Got {len(pairs)} pairs, {len(distractors)} distractors.")

    all_urls = list({p["ref_url"] for p in pairs} | {p["query_url"] for p in pairs}
                     | {d["image_url"] for d in distractors})
    logger.info(f"Downloading {len(all_urls)} images …")
    blobs = download_all(all_urls)

    def load(url):
        data = blobs.get(url)
        if data is None:
            return None
        try:
            return embeddings.load_image(data)
        except Exception:
            return None

    ref_items = []   # (sku, image)
    for p in pairs:
        img = load(p["ref_url"])
        if img is not None:
            ref_items.append((p["sku"], img))
    for d in distractors:
        img = load(d["image_url"])
        if img is not None:
            ref_items.append((d["sku"], img))

    query_items = []  # (true_sku, image)
    for p in pairs:
        img = load(p["query_url"])
        if img is not None:
            query_items.append((p["sku"], img))

    logger.info(f"Reference set: {len(ref_items)} images ({len(pairs)} paired + distractors). "
                f"Query set: {len(query_items)} held-out images.")
    if len(query_items) < 10:
        sys.exit("Too few usable query images (downloads failed?) — can't get a meaningful result.")

    logger.info("Embedding reference set (all poolings, one forward pass) …")
    ref_multi = embeddings.embed_images_multi([img for _, img in ref_items])
    ref_skus = [sku for sku, _ in ref_items]

    logger.info("Embedding query set (all poolings, one forward pass) …")
    query_multi = embeddings.embed_images_multi([img for _, img in query_items])
    query_skus = [sku for sku, _ in query_items]

    print(f"\n{'pooling':<18} {'top1_acc':>10} {'top2_acc':>10} {'mean_margin(correct)':>22}")
    for pooling in POOLINGS:
        ref_vecs = ref_multi[pooling]      # (R, D)
        q_vecs = query_multi[pooling]       # (Q, D)
        sims = q_vecs @ ref_vecs.T           # (Q, R) cosine sim (already L2-normalized)

        top1_hits, top2_hits, margins = 0, 0, []
        for i, true_sku in enumerate(query_skus):
            order = np.argsort(-sims[i])
            top_skus = [ref_skus[j] for j in order[:2]]
            if top_skus[0] == true_sku:
                top1_hits += 1
                margins.append(sims[i, order[0]] - sims[i, order[1]])
            if true_sku in top_skus:
                top2_hits += 1

        n = len(query_skus)
        mean_margin = float(np.mean(margins)) if margins else float("nan")
        print(f"{pooling:<18} {top1_hits / n:>10.3f} {top2_hits / n:>10.3f} {mean_margin:>22.4f}")

    print(f"\n(n_query={len(query_items)}, n_reference={len(ref_items)})")
    print("top1_acc: fraction where a genuinely different photo of the same shoe ranked #1.")
    print("mean_margin(correct): avg similarity gap between #1 and #2 on correct hits — ")
    print("higher means the confidence gate (MIN_MATCH_MARGIN) has more room to work with.")


if __name__ == "__main__":
    main()
