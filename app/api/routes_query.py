"""``POST /query``."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from app.api.deps import ContainerDep
from app.schemas import Citation, QueryRequest, QueryResponse
from llm.client import LLMError

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest, container: ContainerDep) -> QueryResponse:
    """Answer a question with the agentic hybrid-retrieval workflow."""
    s = container.settings
    top_k = body.options.top_k if body.options.top_k is not None else s.top_k
    max_iter = (
        body.options.max_iterations
        if body.options.max_iterations is not None
        else s.max_iterations
    )
    try:
        state = await asyncio.wait_for(
            container.agent.answer(body.query, top_k=top_k, max_iterations=max_iter),
            timeout=s.query_timeout_s,
        )
    except TimeoutError as exc:
        raise HTTPException(504, "query timed out") from exc
    except LLMError as exc:
        raise HTTPException(502, f"LLM backend error: {exc}") from exc

    by_id = {item.chunk_id: item for item in state["items"]}
    citations = [
        Citation(chunk_id=cid, doc_id=by_id[cid].doc_id, text=by_id[cid].text)
        for cid in state["citations"]
        if cid in by_id
    ]
    return QueryResponse(
        answer=state["answer"],
        grounded=state["grounded"],
        insufficient_evidence=state["insufficient"],
        citations=citations,
        graph_context=state["graph_context"],
        retrieved=state["items"],
        trace=state["trace"],
    )
