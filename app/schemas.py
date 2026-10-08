"""Pydantic v2 request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from graph.models import GraphContext
from retrieval.models import RetrievedItem


class QueryOptions(BaseModel):
    """Per-request overrides."""

    top_k: int | None = Field(default=None, ge=1, le=20)
    max_iterations: int | None = Field(default=None, ge=0, le=5)


class QueryRequest(BaseModel):
    """Body of ``POST /query``."""

    query: str = Field(min_length=1, max_length=2000)
    options: QueryOptions = QueryOptions()


class Citation(BaseModel):
    """A source chunk supporting the answer."""

    chunk_id: str
    doc_id: str
    text: str


class QueryResponse(BaseModel):
    """Body returned by ``POST /query``."""

    answer: str
    grounded: bool
    insufficient_evidence: bool
    citations: list[Citation]
    graph_context: GraphContext
    retrieved: list[RetrievedItem]
    trace: list[dict[str, Any]]


class IngestAccepted(BaseModel):
    """Body returned by ``POST /ingest`` (HTTP 202)."""

    job_id: str
    status: str
