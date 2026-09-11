"""
Nearest-neighbor index over reference shoe embeddings.

Plain NumPy matrix-multiply search, not FAISS: at this project's scale (the
sneaker-arbitrage DB's distinct-SKU count — thousands, not millions), a full
brute-force scan (one BLAS matvec) answers a query in low single-digit
milliseconds, same as FAISS's flat index would. FAISS was tried first and
dropped after it turned out to crash on Windows when imported in the same
process as torch (`OMP: Error #15` — the two ship conflicting bundled
OpenMP runtimes; reproducible, not a fluke — see git history/README if this
ever needs revisiting). NumPy has no such conflict and needs no extra native
dependency. Revisit only if the reference catalog grows into the hundreds of
thousands of SKUs, where an approximate index would start to matter.

Vectors are L2-normalized at embed time (app/embeddings.py), so a dot
product IS cosine similarity.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from app.config import settings
from app.models import Candidate

logger = logging.getLogger(__name__)

_VECTORS_FILE = "vectors.npy"
_META_FILE = "metadata.json"


@dataclass
class ReferenceItem:
    sku: str
    name: Optional[str]
    image_url: Optional[str]


class ReferenceIndex:
    def __init__(self, vectors: np.ndarray, items: List[ReferenceItem]):
        self._vectors = vectors   # (N, D) float32, L2-normalized rows
        self._items = items

    @classmethod
    def build(cls, items: List[ReferenceItem], embeddings: np.ndarray) -> "ReferenceIndex":
        if len(items) != embeddings.shape[0]:
            raise ValueError(f"{len(items)} items vs {embeddings.shape[0]} embeddings")
        return cls(embeddings.astype("float32", copy=False), items)

    def save(self, directory: Optional[str] = None):
        d = Path(directory or settings.reference_index_dir)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / _VECTORS_FILE, self._vectors)
        with open(d / _META_FILE, "w", encoding="utf-8") as f:
            json.dump([vars(item) for item in self._items], f)
        logger.info(f"Saved reference index ({len(self._items)} items) to {d}")

    @classmethod
    def load(cls, directory: Optional[str] = None) -> "ReferenceIndex":
        d = Path(directory or settings.reference_index_dir)
        vectors_path, meta_path = d / _VECTORS_FILE, d / _META_FILE
        if not vectors_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"No reference index at {d} — run scripts/build_reference_index.py first."
            )
        vectors = np.load(vectors_path)
        with open(meta_path, encoding="utf-8") as f:
            items = [ReferenceItem(**row) for row in json.load(f)]
        return cls(vectors, items)

    def __len__(self) -> int:
        return len(self._items)

    def search(self, query_vec: np.ndarray, k: Optional[int] = None) -> List[Candidate]:
        """Top-k candidates for one query embedding, sorted by descending
        similarity. Does not itself decide confidence — see
        classify_match() for the identified/ambiguous/no-match gate."""
        k = k or settings.top_k
        n = len(self._items)
        k = min(k, n)
        if k == 0:
            return []
        scores = self._vectors @ query_vec.astype("float32", copy=False)
        # argpartition for O(n) top-k selection, then sort just those k.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        out = []
        for idx in top_idx:
            item = self._items[idx]
            out.append(Candidate(sku=item.sku, name=item.name, similarity=float(scores[idx])))
        return out


def classify_match(candidates: List[Candidate]) -> "tuple[bool, Optional[str]]":
    """Applies the confidence gate (settings.min_match_similarity /
    min_match_margin) to a sorted candidate list. Returns (identified, reason).
    reason is set only when identified=False."""
    if not candidates:
        return False, "no_candidates"
    top = candidates[0]
    if top.similarity < settings.min_match_similarity:
        return False, "no_confident_match"
    if len(candidates) > 1:
        margin = top.similarity - candidates[1].similarity
        if margin < settings.min_match_margin:
            return False, "ambiguous_top_match"
    return True, None
