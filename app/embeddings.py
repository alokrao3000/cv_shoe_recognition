"""
Image -> embedding vector, via DINOv2 (facebook/dinov2-base by default).

DINOv2 is used instead of CLIP because this is fine-grained INSTANCE
retrieval (tell a Jordan 4 "Black Cat" apart from a "Bred" of the same
silhouette), not loose semantic matching — DINOv2's self-supervised training
optimizes for visual detail, whereas CLIP is optimized to align with caption
text and tends to blur similar colorways together.

Pooling: CLS token by default (settings.embedding_pooling). A real weak-
discrimination symptom (an unrelated Cortez nearly tied with the correct Air
Force 1 for a live query) suggested trying mean-pooled patch tokens instead
— the textbook fix for "global embedding blurs local detail". Tested it
properly on a real leave-out task first (scripts/eval_pooling.py: hold out a
genuinely different retailer's photo of the same SKU, check whether it still
ranks #1 against thousands of other candidates) rather than shipping on
assumption, and the textbook fix was wrong here: CLS scored 57.5% top-1
accuracy vs. 22.5% for mean_patch and 51.2% for a CLS+mean_patch concat —
patch tokens apparently carry more background/crop/angle noise than useful
fine-grained signal across different retailers' product photos. Don't switch
this default without re-running that eval on real data. See embed_images_
multi() for computing every pooling variant from one forward pass, which is
how the three were compared without a 3x inference cost.

Runs on GPU automatically if torch sees one (CUDA build installed); falls
back to CPU otherwise. A single image embeds in well under a second on CPU;
embedding a multi-thousand-image reference catalog is the slow part — see
scripts/build_reference_index.py for batching.
"""
import io
import logging
from typing import Dict, List, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from app.config import settings

logger = logging.getLogger(__name__)

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_processor = None

POOLINGS = ("cls", "mean_patch", "cls_mean_concat")


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


def _pool(hidden: torch.Tensor, pooling: str) -> torch.Tensor:
    """hidden: (B, tokens, D) last_hidden_state, token 0 = CLS, rest = patches."""
    if pooling == "cls":
        return hidden[:, 0, :]
    if pooling == "mean_patch":
        return hidden[:, 1:, :].mean(dim=1)
    if pooling == "cls_mean_concat":
        return torch.cat([hidden[:, 0, :], hidden[:, 1:, :].mean(dim=1)], dim=1)
    raise ValueError(f"Unknown pooling {pooling!r} — expected one of {POOLINGS}")


@torch.no_grad()
def embed_images(images: List[Image.Image], batch_size: int = 16,
                  pooling: "str | None" = None) -> np.ndarray:
    """Returns an (N, D) float32 array of L2-normalized embeddings — L2-normalized
    so a plain dot product in app/index.py IS cosine similarity."""
    pooling = pooling or settings.embedding_pooling
    _load()
    out = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = _processor(images=batch, return_tensors="pt").to(_device)
        hidden = _model(**inputs).last_hidden_state
        pooled = _pool(hidden, pooling)
        vecs = torch.nn.functional.normalize(pooled, p=2, dim=1)
        out.append(vecs.cpu().numpy().astype("float32"))
    return np.concatenate(out, axis=0)


def embed_image(image: Image.Image, pooling: "str | None" = None) -> np.ndarray:
    """Single-image convenience wrapper — returns a (D,) vector."""
    return embed_images([image], pooling=pooling)[0]


@torch.no_grad()
def embed_images_multi(images: List[Image.Image], batch_size: int = 16) -> Dict[str, np.ndarray]:
    """Like embed_images(), but computes EVERY pooling variant in POOLINGS from
    the same forward pass — for scripts/eval_pooling.py, so comparing N
    poolings costs one inference pass, not N."""
    _load()
    out = {p: [] for p in POOLINGS}
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = _processor(images=batch, return_tensors="pt").to(_device)
        hidden = _model(**inputs).last_hidden_state
        for p in POOLINGS:
            vecs = torch.nn.functional.normalize(_pool(hidden, p), p=2, dim=1)
            out[p].append(vecs.cpu().numpy().astype("float32"))
    return {p: np.concatenate(chunks, axis=0) for p, chunks in out.items()}
