import io

import numpy as np
import pytest
from PIL import Image

from app import embeddings


def _fake_jpeg_bytes(color=(200, 50, 50)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format="JPEG")
    return buf.getvalue()


def test_load_image_from_bytes():
    img = embeddings.load_image(_fake_jpeg_bytes())
    assert img.mode == "RGB"
    assert img.size == (64, 64)


def test_load_image_converts_palette_mode():
    buf = io.BytesIO()
    Image.new("P", (32, 32)).save(buf, format="PNG")
    img = embeddings.load_image(buf.getvalue())
    assert img.mode == "RGB"


@pytest.mark.slow
def test_embed_image_returns_normalized_vector():
    """Downloads model weights on first run — skipped by default (see
    pytest.ini); run with `pytest -m slow` when you want the real check."""
    img = embeddings.load_image(_fake_jpeg_bytes())
    vec = embeddings.embed_image(img)
    assert vec.ndim == 1
    assert vec.dtype == np.float32
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-3)
