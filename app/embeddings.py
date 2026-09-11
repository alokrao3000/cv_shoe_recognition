"""
Image -> embedding vector, via DINOv2 (facebook/dinov2-base by default).

DINOv2 is used instead of CLIP because this is fine-grained INSTANCE
retrieval (tell a Jordan 4 "Black Cat" apart from a "Bred" of the same
silhouette), not loose semantic matching — DINOv2's self-supervised training
optimizes for visual detail, whereas CLIP is optimized to align with caption
text and tends to blur similar colorways together.

Runs on GPU automatically if torch sees one (CUDA build installed); falls
back to CPU otherwise. A single image embeds in well under a second on CPU;
embedding a multi-thousand-image reference catalog is the slow part — see
scripts/build_reference_index.py for batching.
"""
import io
import logging
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from app.config import settings

logger = logging.getLogger(__name__)

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is not None:
        return
    logger.info(f"Loading embedding model {settings.embedding_model} on {_device} …")
    _processor = AutoImageProcessor.from_pretrained(settings.embedding_model)
    _model = AutoModel.from_pretrained(settings.embedding_model).to(_device).eval()


def load_image(data: Union[bytes, str]) -> Image.Image:
    """Accepts raw image bytes or a filesystem path."""
    if isinstance(data, (bytes, bytearray)):
        img = Image.open(io.BytesIO(data))
    else:
        img = Image.open(data)
    return img.convert("RGB")


@torch.no_grad()
def embed_images(images: List[Image.Image], batch_size: int = 16) -> np.ndarray:
    """Returns an (N, D) float32 array of L2-normalized embeddings — L2-normalized
    so a plain dot product in app/index.py IS cosine similarity."""
    _load()
    out = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = _processor(images=batch, return_tensors="pt").to(_device)
        outputs = _model(**inputs)
        # DINOv2's pooled [CLS] embedding — the standard instance-retrieval
        # representation for this model family.
        pooled = outputs.last_hidden_state[:, 0, :]
        vecs = torch.nn.functional.normalize(pooled, p=2, dim=1)
        out.append(vecs.cpu().numpy().astype("float32"))
    return np.concatenate(out, axis=0)


def embed_image(image: Image.Image) -> np.ndarray:
    """Single-image convenience wrapper — returns a (D,) vector."""
    return embed_images([image])[0]
