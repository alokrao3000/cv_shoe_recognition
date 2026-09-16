"""Resolution cache: same images + same page identity → reuse a prior confident result."""
import hashlib
from datetime import timedelta
from typing import List, Optional
from urllib.parse import urlparse, urlunparse

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Identification, utcnow


def normalize_url(url: str) -> str:
    if not url:
        return ""
    p = urlparse(url.strip())
    return urlunparse((p.scheme.lower(), (p.netloc or "").lower(), p.path.rstrip("/"), "", "", ""))


def cache_key(image_hashes: List[str], product_url: str, title: str) -> str:
    material = "|".join(sorted(image_hashes)) + "|" + normalize_url(product_url) + "|" + " ".join((title or "").lower().split())
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def lookup(session: Session, key: str) -> Optional[Identification]:
    cutoff = utcnow() - timedelta(hours=settings.resolution_cache_ttl_hours)
    stmt = (select(Identification)
            .where(Identification.cache_key == key, Identification.status.in_(("high", "medium")),
                   Identification.created_at >= cutoff)
            .order_by(Identification.created_at.desc()).limit(1))
    return session.execute(stmt).scalar_one_or_none()
