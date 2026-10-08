"""Hybrid retriever: dense + graph, fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from graph.models import GraphContext
from graph.store import GraphStore
from retrieval.dense import DenseRetriever
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.graph import GraphRetriever
from retrieval.models import Mode, Origin, RetrievalResult, RetrievedItem


class HybridRetriever:
    """Runs the selected retrievers and fuses their rankings."""

    def __init__(
        self,
        dense: DenseRetriever,
        graph: GraphRetriever,
        store: GraphStore,
        rrf_k: int = 60,
    ) -> None:
        self._dense = dense
        self._graph = graph
        self._store = store
        self._rrf_k = rrf_k

    async def retrieve(self, query: str, mode: Mode, k: int) -> RetrievalResult:
        """Retrieve the top-``k`` chunks using ``mode``.

        Each retriever contributes up to ``max(2k, 10)`` candidates to the
        fusion step. With a single retriever the fused order equals that
        retriever's order.
        """
        fetch = max(2 * k, 10)
        vector_ids: list[str] = []
        graph_ids: list[str] = []
        context = GraphContext()

        if mode in ("vector", "both"):
            vector_ids = [cid for cid, _ in await self._dense.search(query, fetch)]
        if mode in ("graph", "both"):
            result = self._graph.search(query)
            graph_ids = result.chunk_ids[:fetch]
            context = result.context

        fused = reciprocal_rank_fusion(
            {"vector": vector_ids, "graph": graph_ids}, k=self._rrf_k
        )
        in_vector, in_graph = set(vector_ids), set(graph_ids)

        items: list[RetrievedItem] = []
        for chunk_id, score in fused:
            chunk = self._store.get_chunk(chunk_id)
            if chunk is None:
                continue
            origin: Origin = (
                "both"
                if chunk_id in in_vector and chunk_id in in_graph
                else "vector" if chunk_id in in_vector else "graph"
            )
            items.append(
                RetrievedItem(
                    chunk_id=chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=score,
                    origin=origin,
                )
            )
            if len(items) == k:
                break
        return RetrievalResult(items=items, graph_context=context)
