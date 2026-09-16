"""pgvector cosine search over product_images, max-pooled per product."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Product, ProductImage


@dataclass
class SimilarProduct:
    product_id: int
    style_code: str
    style_code_display: str
    name: str
    brand: str
    model: str
    colorway: str
    gender: str
    size_category: str
    image_url: str
    similarity: float
    matched_image_id: int


def search_similar(session: Session, query_vectors: Sequence[np.ndarray], k: int = 10,
                   brand: Optional[str] = None) -> List[SimilarProduct]:
    """Top-k products by cosine similarity, taking the best hit across all
    query images (max-pool). Optional brand filter (case-insensitive)."""
    best: Dict[int, SimilarProduct] = {}
    for vec in query_vectors:
        v = np.asarray(vec, dtype="float32").tolist()
        distance = ProductImage.embedding.cosine_distance(v)
        stmt = (select(ProductImage.id, ProductImage.product_id, ProductImage.image_url, distance.label("d"), Product)
                .join(Product, Product.id == ProductImage.product_id)
                .order_by(distance).limit(k * 3))
        if brand:
            stmt = stmt.where(Product.brand.ilike(brand))
        for image_id, product_id, image_url, d, product in session.execute(stmt):
            sim = float(1.0 - d)
            cur = best.get(product_id)
            if cur is None or sim > cur.similarity:
                best[product_id] = SimilarProduct(
                    product_id=product_id, style_code=product.style_code,
                    style_code_display=product.style_code_display or product.style_code,
                    name=product.name or "", brand=product.brand or "", model=product.model or "",
                    colorway=product.colorway or "", gender=product.gender or "",
                    size_category=product.size_category or "", image_url=image_url or "",
                    similarity=sim, matched_image_id=image_id,
                )
    return sorted(best.values(), key=lambda s: -s.similarity)[:k]


def similarity_to_product(session: Session, query_vectors: Sequence[np.ndarray], product_id: int) -> Optional[float]:
    """Best cosine similarity between the query images and one product's reference images."""
    rows = session.execute(select(ProductImage.embedding).where(ProductImage.product_id == product_id)).all()
    if not rows:
        return None
    refs = np.asarray([np.asarray(r[0], dtype="float32") for r in rows])
    q = np.asarray([np.asarray(v, dtype="float32") for v in query_vectors])
    return float((q @ refs.T).max())
