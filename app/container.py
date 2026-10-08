"""Dependency container: builds and wires every component once."""

from __future__ import annotations

from dataclasses import dataclass

from agent.workflow import GraphRagAgent
from app.config import Settings
from graph.store import GraphStore, InMemoryGraphStore
from ingestion.jobs import JobStore
from llm.client import LLMClient, OpenAICompatClient
from retrieval.dense import DenseRetriever, InMemoryVectorStore
from retrieval.embeddings import Embedder, build_embedder
from retrieval.graph import GraphRetriever
from retrieval.hybrid import HybridRetriever


@dataclass
class Container:
    """All long-lived components of the running application."""

    settings: Settings
    llm: LLMClient
    embedder: Embedder
    graph_store: GraphStore
    vector_store: InMemoryVectorStore
    retriever: HybridRetriever
    agent: GraphRagAgent
    jobs: JobStore


def build_container(
    settings: Settings,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
) -> Container:
    """Create a container; ``llm`` / ``embedder`` can be injected (tests)."""
    llm = llm or OpenAICompatClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        timeout_s=settings.llm_timeout_s,
    )
    embedder = embedder or build_embedder(settings.embedder, settings.embedding_model)
    graph_store = InMemoryGraphStore()
    vector_store = InMemoryVectorStore()
    retriever = HybridRetriever(
        dense=DenseRetriever(embedder, vector_store),
        graph=GraphRetriever(
            graph_store, settings.graph_hops, settings.graph_entity_limit
        ),
        store=graph_store,
        rrf_k=settings.rrf_k,
    )
    return Container(
        settings=settings,
        llm=llm,
        embedder=embedder,
        graph_store=graph_store,
        vector_store=vector_store,
        retriever=retriever,
        agent=GraphRagAgent(llm, retriever),
        jobs=JobStore(),
    )
