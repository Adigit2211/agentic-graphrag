"""Dense retrieval over an in-memory cosine-similarity index.

v0.1 uses a numpy matrix. A Qdrant-backed store is planned for v0.2.
"""

from __future__ import annotations

import asyncio

import numpy as np

from retrieval.embeddings import Embedder, Vectors


class InMemoryVectorStore:
    """Maps chunk IDs to L2-normalized vectors; ``upsert`` is idempotent."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._index: dict[str, int] = {}
        self._vecs: list[Vectors] = []
        self._matrix: Vectors | None = None

    def __len__(self) -> int:
        return len(self._ids)

    def upsert(self, chunk_id: str, vector: Vectors) -> None:
        """Insert or replace the vector for ``chunk_id``."""
        pos = self._index.get(chunk_id)
        if pos is None:
            self._index[chunk_id] = len(self._ids)
            self._ids.append(chunk_id)
            self._vecs.append(vector)
        else:
            self._vecs[pos] = vector
        self._matrix = None

    def search(self, vector: Vectors, k: int) -> list[tuple[str, float]]:
        """Return the top-``k`` ``(chunk_id, cosine_score)`` pairs."""
        if not self._ids:
            return []
        if self._matrix is None:
            self._matrix = np.vstack(self._vecs)
        scores = self._matrix @ vector
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self._ids[i], float(scores[i])) for i in order]


class DenseRetriever:
    """Embeds the query and searches the vector store."""

    def __init__(self, embedder: Embedder, store: InMemoryVectorStore) -> None:
        self._embedder = embedder
        self._store = store

    async def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return the top-``k`` chunk IDs with similarity scores."""
        vec = await asyncio.to_thread(self._embedder.embed, [query])
        return self._store.search(vec[0], k)
