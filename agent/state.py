"""Typed state carried through the LangGraph workflow."""

from __future__ import annotations

from typing import Any, TypedDict

from graph.models import GraphContext
from retrieval.models import RetrievedItem


class AgentState(TypedDict, total=False):
    """Mutable state of one agent run. Nodes return partial updates."""

    question: str
    current_query: str
    top_k: int
    max_iterations: int
    iteration: int
    route: str
    tried_routes: list[str]
    fallback_used: bool
    items: list[RetrievedItem]
    graph_context: GraphContext
    verdict: dict[str, Any]
    answer: str
    citations: list[str]
    grounded: bool
    insufficient: bool
    trace: list[dict[str, Any]]


def initial_state(question: str, top_k: int, max_iterations: int) -> AgentState:
    """Build the starting state for a question."""
    return AgentState(
        question=question,
        current_query=question,
        top_k=top_k,
        max_iterations=max_iterations,
        iteration=0,
        route="both",
        tried_routes=[],
        fallback_used=False,
        items=[],
        graph_context=GraphContext(),
        verdict={},
        answer="",
        citations=[],
        grounded=False,
        insufficient=False,
        trace=[],
    )
