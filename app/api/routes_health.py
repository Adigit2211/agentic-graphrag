"""``GET /health``: reports each backing component separately."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.deps import ContainerDep

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(container: ContainerDep) -> JSONResponse:
    """Return 200 if every component is healthy, else 503.

    In v0.1 the graph and vector stores are in-process, so they are always
    reachable; the LLM backend is the only network dependency.
    """
    llm_ok = await container.llm.ping()
    body: dict[str, Any] = {
        "status": "ok" if llm_ok else "degraded",
        "llm": {"ok": llm_ok, "model": container.settings.llm_model},
        "graph_store": {
            "ok": True,
            "backend": "in-memory",
            **container.graph_store.stats(),
        },
        "vector_store": {
            "ok": True,
            "backend": "in-memory",
            "vectors": len(container.vector_store),
        },
    }
    return JSONResponse(body, status_code=200 if llm_ok else 503)
