"""Graph storage: a protocol and an in-memory implementation.

v0.1 ships only the in-memory store. A Neo4j implementation of the same
protocol is planned for v0.2 (see README, "Roadmap").
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

from graph.models import (
    ChunkRecord,
    EntityRecord,
    GraphContext,
    GraphEdge,
    GraphNode,
)


class GraphStore(Protocol):
    """Operations the ingestion pipeline and graph retriever rely on."""

    def add_chunk(self, chunk: ChunkRecord) -> None:
        """Insert a chunk (idempotent)."""
        ...

    def add_entity(self, entity: EntityRecord) -> None:
        """Insert an entity (idempotent)."""
        ...

    def add_mention(self, entity_id: str, chunk_id: str) -> None:
        """Record that a chunk mentions an entity (idempotent)."""
        ...

    def add_relation(
        self, subject_id: str, predicate: str, object_id: str, chunk_id: str
    ) -> None:
        """Record a relation with its source chunk (idempotent)."""
        ...

    def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        """Look up a chunk by ID."""
        ...

    def link_entities(self, normalized_query: str) -> list[str]:
        """Return IDs of entities whose names occur in the normalized query."""
        ...

    def expand(self, seed_ids: Sequence[str], hops: int, limit: int) -> dict[str, int]:
        """Breadth-first expansion; returns ``entity_id -> hop distance``."""
        ...

    def chunks_for_entities(self, distances: Mapping[str, int]) -> list[str]:
        """Rank chunks that mention the given entities."""
        ...

    def subgraph(self, entity_ids: Iterable[str]) -> GraphContext:
        """Return nodes and edges induced by ``entity_ids``."""
        ...

    def stats(self) -> dict[str, int]:
        """Return counts of chunks, entities and relations."""
        ...


class InMemoryGraphStore:
    """Dict/set backed graph. Every write is idempotent by construction."""

    def __init__(self) -> None:
        self._chunks: dict[str, ChunkRecord] = {}
        self._entities: dict[str, EntityRecord] = {}
        self._mentions: dict[str, set[str]] = defaultdict(set)
        self._edges: dict[tuple[str, str, str], set[str]] = {}
        self._adj: dict[str, set[str]] = defaultdict(set)

    def add_chunk(self, chunk: ChunkRecord) -> None:
        self._chunks[chunk.id] = chunk

    def add_entity(self, entity: EntityRecord) -> None:
        self._entities.setdefault(entity.id, entity)

    def add_mention(self, entity_id: str, chunk_id: str) -> None:
        self._mentions[entity_id].add(chunk_id)

    def add_relation(
        self, subject_id: str, predicate: str, object_id: str, chunk_id: str
    ) -> None:
        self._edges.setdefault((subject_id, predicate, object_id), set()).add(chunk_id)
        self._adj[subject_id].add(object_id)
        self._adj[object_id].add(subject_id)

    def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        return self._chunks.get(chunk_id)

    def link_entities(self, normalized_query: str) -> list[str]:
        padded = f" {normalized_query} "
        hits = [e for e in self._entities.values() if f" {e.norm_name} " in padded]
        # Longest first, then drop names contained in an already chosen longer
        # name (so "labs" is not linked when "aurora labs" matched).
        hits.sort(key=lambda e: (-len(e.norm_name), e.norm_name))
        chosen: list[EntityRecord] = []
        for entity in hits:
            needle = f" {entity.norm_name} "
            if not any(needle in f" {c.norm_name} " for c in chosen):
                chosen.append(entity)
        return [e.id for e in chosen]

    def expand(self, seed_ids: Sequence[str], hops: int, limit: int) -> dict[str, int]:
        dist: dict[str, int] = {s: 0 for s in seed_ids if s in self._entities}
        frontier = sorted(dist)
        for hop in range(1, hops + 1):
            nxt: list[str] = []
            for node in frontier:
                for neighbour in sorted(self._adj.get(node, ())):
                    if neighbour not in dist and len(dist) < limit:
                        dist[neighbour] = hop
                        nxt.append(neighbour)
            frontier = nxt
        return dist

    def chunks_for_entities(self, distances: Mapping[str, int]) -> list[str]:
        best: dict[str, int] = {}
        count: dict[str, int] = defaultdict(int)
        for entity_id, d in distances.items():
            for chunk_id in self._mentions.get(entity_id, ()):
                best[chunk_id] = min(best.get(chunk_id, d), d)
                count[chunk_id] += 1
        return sorted(best, key=lambda c: (best[c], -count[c], c))

    def subgraph(self, entity_ids: Iterable[str]) -> GraphContext:
        ids = {i for i in entity_ids if i in self._entities}
        nodes = [
            GraphNode(id=e.id, name=e.name, type=e.type)
            for e in sorted((self._entities[i] for i in ids), key=lambda e: e.norm_name)
        ]
        edges = [
            GraphEdge(source=s, predicate=p, target=o, chunk_ids=sorted(chunks))
            for (s, p, o), chunks in sorted(self._edges.items())
            if s in ids and o in ids
        ]
        return GraphContext(nodes=nodes, edges=edges)

    def stats(self) -> dict[str, int]:
        return {
            "chunks": len(self._chunks),
            "entities": len(self._entities),
            "relations": len(self._edges),
        }
