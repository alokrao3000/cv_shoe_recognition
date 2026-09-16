from datetime import datetime, timezone
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import settings


def utcnow() -> datetime:
    """Naive UTC (the DateTime columns are timezone-naive)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Product(Base):
    """One sneaker release, keyed by its normalized style code."""
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(primary_key=True)
    style_code: Mapped[str] = mapped_column(String(64), unique=True, index=True)   # normalized (A-Z0-9 only)
    style_code_display: Mapped[Optional[str]] = mapped_column(String(64))           # as printed, e.g. DD1391-100
    brand: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    model: Mapped[Optional[str]] = mapped_column(String(200), index=True)
    sub_model: Mapped[Optional[str]] = mapped_column(String(200))
    colorway: Mapped[Optional[str]] = mapped_column(String(300))
    name: Mapped[Optional[str]] = mapped_column(String(500))
    gender: Mapped[Optional[str]] = mapped_column(String(30))
    size_category: Mapped[Optional[str]] = mapped_column(String(30))
    release_date: Mapped[Optional[str]] = mapped_column(String(20))
    retail_price: Mapped[Optional[float]] = mapped_column(Float)
    stockx_product_id: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    stockx_url_key: Mapped[Optional[str]] = mapped_column(String(300))
    source: Mapped[str] = mapped_column(String(40), default="unknown")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    images: Mapped[list["ProductImage"]] = relationship(back_populates="product", cascade="all, delete-orphan")


class ProductImage(Base):
    """A reference photo of a product with its DINOv2 embedding."""
    __tablename__ = "product_images"

    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    image_url: Mapped[Optional[str]] = mapped_column(String(1000))
    image_hash: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(40))          # retailer | web | user_confirmed | stockx_seed | query
    embedding = mapped_column(Vector(settings.embedding_dim))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    product: Mapped[Product] = relationship(back_populates="images")

    __table_args__ = (UniqueConstraint("product_id", "image_hash", name="uq_product_image_hash"),)


class Identification(Base):
    """One identification attempt: inputs, the full evidence graph, and the outcome."""
    __tablename__ = "identifications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)   # queued|running|high|medium|low|unresolved|error
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    review_required: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    input_kind: Mapped[str] = mapped_column(String(20))       # image | url | mixed
    input_url: Mapped[Optional[str]] = mapped_column(String(1000))
    input_domain: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    input_title: Mapped[Optional[str]] = mapped_column(String(500))
    input_description: Mapped[Optional[str]] = mapped_column(Text)
    image_hashes: Mapped[list] = mapped_column(JSON, default=list)
    cache_key: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    image_paths: Mapped[list] = mapped_column(JSON, default=list)   # stored copies for the review UI

    selected_style_code: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    selected_product_id: Mapped[Optional[int]] = mapped_column(ForeignKey("products.id"))
    canonical_product: Mapped[Optional[dict]] = mapped_column(JSON)
    stockx: Mapped[Optional[dict]] = mapped_column(JSON)
    candidates: Mapped[list] = mapped_column(JSON, default=list)
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)
    failure_codes: Mapped[list] = mapped_column(JSON, default=list)
    stages: Mapped[list] = mapped_column(JSON, default=list)
    identification_method: Mapped[list] = mapped_column(JSON, default=list)
    error: Mapped[Optional[str]] = mapped_column(Text)

    decisions: Mapped[list["ReviewDecision"]] = relationship(back_populates="identification",
                                                             cascade="all, delete-orphan")


class ReviewDecision(Base):
    """A human verdict on an identification — becomes labeled training data."""
    __tablename__ = "review_decisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    identification_id: Mapped[str] = mapped_column(ForeignKey("identifications.id"), index=True)
    action: Mapped[str] = mapped_column(String(20))           # confirm | reject | select | manual_sku
    chosen_style_code: Mapped[Optional[str]] = mapped_column(String(64))
    system_style_code: Mapped[Optional[str]] = mapped_column(String(64))
    system_confidence: Mapped[Optional[float]] = mapped_column(Float)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    identification: Mapped[Identification] = relationship(back_populates="decisions")


class StockXCache(Base):
    """TTL cache of StockX API responses — protects the daily request budget."""
    __tablename__ = "stockx_cache"

    key: Mapped[str] = mapped_column(String(300), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


Index("ix_product_images_embedding_hnsw", ProductImage.embedding,
      postgresql_using="hnsw", postgresql_with={"m": 16, "ef_construction": 64},
      postgresql_ops={"embedding": "vector_cosine_ops"})
