"""
Builds data/eval/cases.jsonl from the sneaker-arbitrage DB: SKUs that have
photos from two or more different retailers. One photo (plus that
retailer's product URL/title) becomes the query; the others are what the
seeded reference index knows. This measures the real task — recognising
the same release across a genuinely different photo — not memorisation.

Hand-curated cases (collabs, GS vs adult, similar colorways, poor photos)
live in data/eval/curated.jsonl and are merged in.

Usage:
    python scripts/build_eval_set.py [--pairs 100] [--seed 42]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2
import psycopg2.extras

from app.config import settings  # noqa: E402

OUT = Path(settings.data_dir) / "eval" / "cases.jsonl"
CURATED = Path(settings.data_dir) / "eval" / "curated.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    conn = psycopg2.connect(settings.arbitrage_database_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT sku, array_agg(DISTINCT image_url) AS urls,
                       array_agg(DISTINCT product_url) AS product_urls,
                       array_agg(DISTINCT name) AS names,
                       count(DISTINCT supplier_id) AS suppliers
                FROM supplier_products
                WHERE sku IS NOT NULL AND sku != '' AND image_url IS NOT NULL AND image_url != ''
                GROUP BY sku HAVING count(DISTINCT supplier_id) >= 2
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    cases = []
    for r in rows[: args.pairs]:
        urls = [u for u in r["urls"] if u]
        if len(urls) < 2:
            continue
        query_url = urls[-1]
        tags = ["cross_retailer"]
        name = (r["names"] or [""])[0] or ""
        low = name.lower()
        if "(gs)" in low or "grade school" in low or "kids" in low:
            tags.append("gs")
        if "women" in low or "wmns" in low or "(w)" in low:
            tags.append("womens")
        cases.append({"image_url": query_url, "title": name, "product_url": (r["product_urls"] or [""])[-1] or "",
                      "expected_style_code": r["sku"], "tags": tags})

    if CURATED.exists():
        for line in CURATED.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                cases.append(json.loads(line))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
