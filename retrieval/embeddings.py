"""Text embedders: sentence-transformers (default) and a hashing fallback."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Vectors = NDArray[np.float32]


class Embedder(Protocol):
    """Maps texts to L2-normalized float32 vectors, shape ``(n, dim)``."""

    def embed(self, texts: Sequence[str]) -> Vectors:
        """Embed a batch of texts."""
        ...


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder.

    Needs no model download, so it is used in tests and for offline smoke runs.
    It captures lexical overlap only, NOT semantic similarity.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> Vectors:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                out[row, int.from_bytes(digest[:8], "big") % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.asarray(out / norms, dtype=np.float32)


class SentenceTransformerEmbedder:
    """Dense embeddings from a sentence-transformers model."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(
                "sentence-transformers is not installed. Run "
                "`pip install -e '.[embeddings]'` or set EMBEDDER=hashing."
            ) from exc
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> Vectors:
        vecs = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vecs, dtype=np.float32)


def build_embedder(kind: str, model_name: str) -> Embedder:
    """Create the embedder selected by configuration."""
    if kind == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(model_name)
