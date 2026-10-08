"""Retrieval result models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from graph.models import GraphContext

Mode = Literal["graph", "vector", "both"]
Origin = Literal["graph", "vector", "both"]


class RetrievedItem(BaseModel):
    """A retrieved chunk with provenance and its retrieval origin."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    origin: Origin


class RetrievalResult(BaseModel):
    """Fused retrieval output."""

    items: list[RetrievedItem] = []
    graph_context: GraphContext = GraphContext()
