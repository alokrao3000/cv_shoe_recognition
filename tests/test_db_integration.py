"""Runs against the real Postgres + pgvector (docker compose up -d postgres);
skipped automatically when it's unreachable."""
import numpy as np
import pytest

from app.config import settings
from app.db.session import db_available

pytestmark = pytest.mark.integration

if not db_available():
    pytest.skip("database unavailable", allow_module_level=True)

from app.db import repo                      # noqa: E402
from app.db.models import Product           # noqa: E402
from app.db.session import db_session, init_db   # noqa: E402
from app.db.vector_search import search_similar, similarity_to_product   # noqa: E402


def unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(settings.embedding_dim).astype("float32")
    return v / np.linalg.norm(v)


@pytest.fixture(autouse=True)
def _cleanup():
    init_db()
    yield
    with db_session() as s:
        for code in ("ZZTEST001", "ZZTEST002"):
            p = repo.get_product_by_code(s, code)
            if p is not None:
                s.delete(p)


def test_upsert_add_image_and_vector_search():
    a, b = unit(1), unit(2)
    with db_session() as s:
        p1 = repo.upsert_product(s, "ZZTEST-001", name="Test Shoe One White/Black", brand="Nike", source="test")
        p2 = repo.upsert_product(s, "ZZTEST-002", name="Test Shoe Two Black/White", brand="Nike", source="test")
        repo.add_product_image(s, p1, a, image_url="http://x/a.jpg", source="test")
        repo.add_product_image(s, p2, b, image_url="http://x/b.jpg", source="test")
        # duplicate bytes/url are not stored twice
        repo.add_product_image(s, p1, a, image_url="http://x/a.jpg", source="test")
        assert repo.product_image_count(s, p1.id) == 1

    q = a * 0.9 + unit(3) * 0.1
    q /= np.linalg.norm(q)
    with db_session() as s:
        hits = search_similar(s, [q], k=5)
        codes = [h.style_code for h in hits]
        assert codes[0] == "ZZTEST001"
        assert hits[0].similarity > 0.85
        p1 = repo.get_product_by_code(s, "ZZTEST001")
        assert similarity_to_product(s, [q], p1.id) == pytest.approx(hits[0].similarity, abs=1e-4)
        assert similarity_to_product(s, [q], -1) is None


def test_upsert_prefers_stockx_metadata_and_keeps_display():
    with db_session() as s:
        p = repo.upsert_product(s, "ZZTEST-001", name="Retailer Name", colorway="", source="retailer", display="ZZTEST-001")
        assert p.style_code_display == "ZZTEST-001" and p.name == "Retailer Name"
        repo.upsert_product(s, "ZZTEST001", name="StockX Name", colorway="White/Black", stockx_product_id="sx1",
                            stockx_url_key="stockx-name")
        p = repo.get_product_by_code(s, "ZZTEST001")
        assert p.name == "StockX Name" and p.colorway == "White/Black" and p.stockx_product_id == "sx1"
        assert p.style_code_display == "ZZTEST-001"
