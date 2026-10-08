"""Graph retrieval: entity linking, bounded traversal, chunk ranking.

No LLM-generated queries are involved: linking is a deterministic phrase match
and traversal is a fixed breadth-first expansion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from graph.models import GraphContext
from graph.naming import normalize_name
from graph.store import GraphStore


@dataclass(frozen=True)
class GraphResult:
    """Outcome of one graph retrieval."""

    chunk_ids: list[str] = field(default_factory=list)
    context: GraphContext = field(default_factory=GraphContext)
    linked_entities: list[str] = field(default_factory=list)


class GraphRetriever:
    """Retrieve chunks reachable from entities mentioned in the query."""

    def __init__(self, store: GraphStore, hops: int, entity_limit: int) -> None:
        self._store = store
        self._hops = hops
        self._limit = entity_limit

    def search(self, query: str) -> GraphResult:
        """Link entities in ``query`` and collect chunks within ``hops`` hops.

        Returns an empty result when no known entity appears in the query.
        """
        seeds = self._store.link_entities(normalize_name(query))
        if not seeds:
            return GraphResult()
        distances = self._store.expand(seeds, self._hops, self._limit)
        return GraphResult(
            chunk_ids=self._store.chunks_for_entities(distances),
            context=self._store.subgraph(distances),
            linked_entities=seeds,
        )
