"""
Widens the product catalog beyond what retailers happened to scrape by
enumerating StockX's catalog/search over curated brand/silhouette terms
(an official, rate-limited API call — counts against the daily budget),
then optionally sourcing a reference photo per new product via image search.

Usage:
    python scripts/seed_stockx_catalog.py [--pages 2] [--images 200] [--queries "Nike Dunk Low" ...]
"""
import argparse
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from sqlalchemy import select

from app.db import repo                                              # noqa: E402
from app.db.models import Product, ProductImage                       # noqa: E402
from app.db.session import db_session, init_db                       # noqa: E402
from app.extraction.normalize import parse_title                     # noqa: E402
from app.extraction.page_extractor import fetch_bytes                # noqa: E402
from app.search.provider import get_provider                         # noqa: E402
from app.search.web_discovery import find_reference_images           # noqa: E402
from app.stockx.catalog import StockXCatalog                         # noqa: E402
from app.vision import embeddings                                    # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED_QUERIES = [
    "Nike Air Force 1", "Nike Air Force 1 Low", "Nike Air Force 1 High", "Nike Air Force 1 Shadow",
    "Air Jordan 1", "Air Jordan 1 Low", "Air Jordan 1 High", "Air Jordan 1 Mid", "Air Jordan 2", "Air Jordan 3",
    "Air Jordan 4", "Air Jordan 5", "Air Jordan 6", "Air Jordan 7", "Air Jordan 9", "Air Jordan 11",
    "Air Jordan 12", "Air Jordan 13", "Air Jordan 14", "Travis Scott Jordan", "Off-White Jordan", "Union Jordan",
    "Fragment Jordan", "Nike Dunk Low", "Nike Dunk High", "Nike SB Dunk Low", "Nike SB Dunk High",
    "StrangeLove Dunk", "Off-White Dunk", "Chunky Dunky", "Travis Scott Dunk", "Parra Dunk", "Nike Air Max 1",
    "Nike Air Max 90", "Nike Air Max 95", "Nike Air Max 97", "Nike Air Max 98", "Nike Air Max 270",
    "Nike Air Max Plus", "Nike Air Max 720", "Nike Cortez", "Nike Blazer Mid", "Nike Blazer Low", "Nike Vapormax",
    "Nike P-6000", "Nike Shox", "Nike Vomero 5", "Nike Kobe", "Nike LeBron",
    "Adidas Yeezy 350", "Adidas Yeezy 500", "Adidas Yeezy 700", "Adidas Yeezy 380", "Adidas Yeezy 450",
    "Adidas Yeezy Slide", "Adidas Yeezy Foam Runner", "Adidas Samba", "Adidas Gazelle", "Adidas Superstar",
    "Adidas Stan Smith", "Adidas Campus", "Adidas Ultraboost", "Adidas NMD", "Adidas Forum Low",
    "Adidas Forum High", "Adidas Handball Spezial", "Adidas SL 72",
    "New Balance 550", "New Balance 574", "New Balance 990", "New Balance 991", "New Balance 992",
    "New Balance 993", "New Balance 996", "New Balance 997", "New Balance 2002R", "New Balance 327",
    "New Balance 9060", "New Balance 1906R", "New Balance 530",
    "Asics Gel-Kayano 14", "Asics Gel-1130", "Asics Gel-NYC", "Asics Gel-Lyte III", "Asics Gel-1090", "Asics GT-2160",
    "Converse Chuck 70", "Converse Chuck Taylor", "Vans Old Skool", "Vans Sk8-Hi", "Vans Authentic",
    "Puma Suede", "Puma RS-X", "Puma Palermo", "Puma Speedcat", "Reebok Classic Leather", "Reebok Club C",
    "Reebok Question", "Salomon XT-6", "Salomon XT-4", "On Cloudmonster", "On Cloud 5", "Hoka Bondi",
    "Hoka Clifton", "Under Armour Curry", "Crocs Classic Clog", "Timberland 6 Inch Boot", "UGG Tasman",
]


def seed_products(catalog: StockXCatalog, queries, pages: int) -> int:
    added = 0
    for q in queries:
        for page in range(1, pages + 1):
            try:
                products = catalog._get_client().search_products(q, page_size=50) if page == 1 else \
                    [p for p in (catalog._get_client()._to_product(x) for x in catalog._get_client()._search(q, 50, page)) if p]
            except Exception:
                logger.exception(f"catalog/search failed for {q!r} p{page}")
                break
            with db_session() as s:
                for p in products:
                    if not p.style_codes or "sneaker" not in (p.product_type or "sneakers").lower() and "shoe" not in (p.product_type or "").lower():
                        continue
                    parsed = parse_title(p.title, p.brand)
                    before = repo.get_product_by_code(s, p.style_codes[0]) is not None
                    repo.upsert_product(s, p.style_codes[0], name=p.title, brand=parsed.brand or p.brand, model=parsed.model,
                                        sub_model=parsed.sub_model, colorway=p.colorway, gender=(p.gender or "").lower(),
                                        size_category=parsed.size_category if parsed.size_category != "unknown" else "",
                                        release_date=p.release_date, retail_price=p.retail_price,
                                        stockx_product_id=p.product_id, stockx_url_key=p.url_key, source="stockx_seed",
                                        display=p.style_id.split("/")[0].strip())
                    added += 0 if before else 1
            if len(products) < 50:
                break
        logger.info(f"{q!r}: catalog now +{added} new products")
    return added


def seed_images(max_images: int) -> int:
    provider = get_provider()
    if not provider.available:
        logger.warning("No search provider configured (SERPER_API_KEY) — skipping image sourcing.")
        return 0
    with db_session() as s:
        with_images = {r[0] for r in s.execute(select(ProductImage.product_id).distinct())}
        targets = [p for p in s.execute(select(Product)).scalars() if p.id not in with_images]
    random.shuffle(targets)
    targets = targets[:max_images]
    logger.info(f"Sourcing photos for {len(targets)} products without a reference image …")
    added = 0
    with httpx.Client() as http:
        for i, p in enumerate(targets, 1):
            query = f"{p.name or p.model} {p.style_code_display or p.style_code}".strip()
            for img in find_reference_images(query, provider, num=6):
                data = fetch_bytes(img.image_url, http)
                if not data:
                    continue
                try:
                    pil = embeddings.load_image(data)
                except Exception:
                    continue
                vec = embeddings.embed_image(pil)
                with db_session() as s:
                    product = s.get(Product, p.id)
                    repo.add_product_image(s, product, vec, image_url=img.image_url, image_bytes=data, source="web")
                added += 1
                break
            if i % 25 == 0:
                logger.info(f"{i}/{len(targets)} — {added} photos added")
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=2, help="catalog/search pages (50 results each) per query")
    ap.add_argument("--images", type=int, default=0, help="Max products to source a reference photo for")
    ap.add_argument("--queries", nargs="*", default=None, help="Override the seed query list")
    args = ap.parse_args()

    init_db()
    catalog = StockXCatalog()
    if not catalog.available:
        sys.exit("StockX API not configured — set STOCKX_* in .env and run scripts/stockx_auth.py")
    try:
        n = seed_products(catalog, args.queries or SEED_QUERIES, args.pages)
    finally:
        catalog.close()
    logger.info(f"Catalog seeding done: {n} new products.")
    if args.images:
        logger.info(f"Image sourcing done: {seed_images(args.images)} photos.")


if __name__ == "__main__":
    main()
