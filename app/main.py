"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import routes_health, routes_ingest, routes_query
from app.config import get_settings
from app.container import Container, build_container
from app.logging_config import configure_logging


def create_app(container: Container | None = None) -> FastAPI:
    """Build the app. Pass ``container`` to inject fakes (used by tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = get_settings()
        configure_logging(settings.log_level)
        active = container or build_container(settings)
        app.state.container = active
        yield
        await active.llm.aclose()

    app = FastAPI(
        title="Agentic GraphRAG",
        version="0.1.0",
        description="Hybrid graph + vector RAG with a LangGraph agent loop.",
        lifespan=lifespan,
    )
    if container is not None:
        app.state.container = container
    app.include_router(routes_health.router)
    app.include_router(routes_ingest.router)
    app.include_router(routes_query.router)
    return app


app = create_app()
