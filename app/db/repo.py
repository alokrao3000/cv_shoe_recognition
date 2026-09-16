"""Small repository helpers shared by the pipeline, review flow and scripts."""
import hashlib
from typing import Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Product, ProductImage
from app.extraction.normalize import parse_title
from app.extraction.sku_detector import normalize_style_code


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_product_by_code(session: Session, style_code: str) -> Optional[Product]:
    norm = normalize_style_code(style_code)
    if not norm:
        return None
    return session.execute(select(Product).where(Product.style_code == norm)).scalar_one_or_none()


def upsert_product(session: Session, style_code: str, *, name: str = "", brand: str = "", model: str = "",
                   sub_model: str = "", colorway: str = "", gender: str = "", size_category: str = "",
                   release_date: str = "", retail_price: Optional[float] = None,
                   stockx_product_id: str = "", stockx_url_key: str = "", source: str = "unknown",
                   display: str = "") -> Product:
    """Creates or fills in missing fields (never overwrites a non-empty field
    with an empty one; StockX metadata wins over scraped text)."""
    norm = normalize_style_code(style_code)
    product = get_product_by_code(session, norm)
    if product is None:
        product = Product(style_code=norm, style_code_display=display or style_code, source=source)
        session.add(product)
    parsed = parse_title(name, brand) if name else None
    fields = {
        "name": name, "brand": brand or (parsed.brand if parsed else ""),
        "model": model or (parsed.model if parsed else ""), "sub_model": sub_model or (parsed.sub_model if parsed else ""),
        "colorway": colorway, "gender": gender or (parsed.gender if parsed and parsed.gender != "unknown" else ""),
        "size_category": size_category or (parsed.size_category if parsed and parsed.size_category != "unknown" else ""),
        "release_date": release_date, "retail_price": retail_price,
        "stockx_product_id": stockx_product_id, "stockx_url_key": stockx_url_key,
    }
    stockx_authoritative = bool(stockx_product_id)
    for key, val in fields.items():
        if val in ("", None):
            continue
        current = getattr(product, key)
        if current in ("", None) or (stockx_authoritative and key in ("name", "colorway", "gender", "release_date",
                                                                       "retail_price", "brand", "stockx_product_id",
                                                                       "stockx_url_key")):
            setattr(product, key, val)
    if display and not product.style_code_display:
        product.style_code_display = display
    session.flush()
    return product


def add_product_image(session: Session, product: Product, embedding: np.ndarray, *, image_url: str = "",
                      image_bytes: Optional[bytes] = None, source: str = "web") -> Optional[ProductImage]:
    """Adds a reference image unless the same bytes are already stored for this product."""
    image_hash = sha256_bytes(image_bytes) if image_bytes else sha256_bytes(image_url.encode())
    existing = session.execute(
        select(ProductImage).where(ProductImage.product_id == product.id, ProductImage.image_hash == image_hash)
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    img = ProductImage(product_id=product.id, image_url=image_url, image_hash=image_hash, source=source,
                       embedding=np.asarray(embedding, dtype="float32").tolist())
    session.add(img)
    session.flush()
    return img


def product_image_count(session: Session, product_id: int) -> int:
    return session.execute(select(ProductImage.id).where(ProductImage.product_id == product_id)).all().__len__()
