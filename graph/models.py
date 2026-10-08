"""Data models shared by graph storage and retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel


@dataclass(frozen=True)
class ChunkRecord:
    """A text chunk of a source document."""

    id: str
    doc_id: str
    idx: int
    text: str


@dataclass(frozen=True)
class EntityRecord:
    """A normalized entity node."""

    id: str
    name: str
    norm_name: str
    type: str


class GraphNode(BaseModel):
    """Entity node as returned by the API."""

    id: str
    name: str
    type: str


class GraphEdge(BaseModel):
    """Directed relation between two entities, with provenance."""

    source: str
    predicate: str
    target: str
    chunk_ids: list[str]


class GraphContext(BaseModel):
    """Sub-graph that was traversed to answer a query."""

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
