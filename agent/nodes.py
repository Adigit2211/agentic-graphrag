"""LangGraph node implementations.

Every node returns a partial state update and appends one event to ``trace``.
Router, grader and rewriter degrade gracefully (rule-based fallbacks) when the
LLM output is invalid or the backend errors; the generator does not, and lets
``LLMError`` propagate so the API can report a backend failure.
"""

from __future__ import annotations

import time
from typing import Any

from agent.grading import fallback_route, lexical_support
from agent.router_rules import as_mode, parse_route, rule_route
from agent.state import AgentState
from llm.client import LLMClient, LLMError
from llm.json_utils import extract_json_object
from retrieval.hybrid import HybridRetriever
from retrieval.models import Mode, RetrievedItem

INSUFFICIENT_MESSAGE = (
    "Insufficient evidence: the retrieved context did not support a reliable "
    "answer to this question."
)

ROUTE_SYSTEM = (
    "[task:route]\n"
    "Pick the best retrieval strategy for the question. Reply with ONLY JSON: "
    '{"route": "graph" | "vector" | "both"}.\n'
    "graph = relationships between named entities; vector = answerable from "
    "the wording of a single passage; both = multi-hop or unsure."
)
GRADE_SYSTEM = (
    "[task:grade]\n"
    "Decide whether the context passages contain the information needed to "
    'answer the question. Reply with ONLY JSON: {"relevant": true|false, '
    '"reason": "<one sentence>"}.'
)
REWRITE_SYSTEM = (
    "[task:rewrite]\n"
    "The previous retrieval did not find enough evidence. Rewrite the question "
    "as a standalone, keyword-rich search query. Reply with ONLY JSON: "
    '{"query": "<rewritten query>"}.'
)
GENERATE_SYSTEM = (
    "[task:generate]\n"
    "Answer the question using ONLY the context passages. Each passage starts "
    "with its ID in square brackets. Reply with ONLY JSON: "
    '{"answerable": true|false, "answer": "<short answer>", '
    '"citations": ["<passage id>", ...]}. '
    'If the context does not support an answer, set "answerable" to false.'
)


def render_context(items: list[RetrievedItem]) -> str:
    """Format retrieved items as ``[chunk_id] text`` blocks for prompts."""
    return "\n\n".join(f"[{i.chunk_id}] {i.text}" for i in items)


def _trace(
    state: AgentState, node: str, started: float, **details: Any
) -> list[dict[str, Any]]:
    event = {
        "node": node,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        **details,
    }
    return [*state.get("trace", []), event]


class AgentNodes:
    """Holds dependencies and exposes one coroutine per workflow node."""

    def __init__(self, llm: LLMClient, retriever: HybridRetriever) -> None:
        self._llm = llm
        self._retriever = retriever

    async def route_query(self, state: AgentState) -> dict[str, Any]:
        """Choose graph / vector / both (LLM first, rules as fallback)."""
        started = time.perf_counter()
        question = state["question"]
        source = "llm"
        try:
            route: Mode = parse_route(
                await self._llm.complete(ROUTE_SYSTEM, question, json_mode=True)
            )
        except (LLMError, ValueError):
            route = rule_route(question)
            source = "rules"
        return {
            "route": route,
            "tried_routes": [route],
            "trace": _trace(state, "route_query", started, route=route, source=source),
        }

    async def retrieve_context(self, state: AgentState) -> dict[str, Any]:
        """Run the selected retriever(s) for the current query."""
        started = time.perf_counter()
        mode: Mode = as_mode(state["route"])
        result = await self._retriever.retrieve(
            state["current_query"], mode, state["top_k"]
        )
        return {
            "items": result.items,
            "graph_context": result.graph_context,
            "trace": _trace(
                state,
                "retrieve_context",
                started,
                route=mode,
                query=state["current_query"],
                retrieved=len(result.items),
                graph_nodes=len(result.graph_context.nodes),
            ),
        }

    async def grade_context(self, state: AgentState) -> dict[str, Any]:
        """Judge whether the context can answer the question."""
        started = time.perf_counter()
        items = state["items"]
        if not items:
            verdict: dict[str, Any] = {
                "relevant": False,
                "reason": "no passages retrieved",
                "source": "rules",
            }
        else:
            prompt = (
                f"QUESTION: {state['question']}\n\nCONTEXT:\n{render_context(items)}"
            )
            try:
                obj = extract_json_object(
                    await self._llm.complete(GRADE_SYSTEM, prompt, json_mode=True)
                )
                relevant = obj.get("relevant")
                if not isinstance(relevant, bool):
                    raise ValueError("'relevant' must be a boolean")
                verdict = {
                    "relevant": relevant,
                    "reason": str(obj.get("reason", "")),
                    "source": "llm",
                }
            except (LLMError, ValueError):
                support = lexical_support(state["question"], items)
                verdict = {
                    "relevant": support >= 0.5,
                    "reason": f"lexical support {support:.2f} (LLM grader unusable)",
                    "source": "rules",
                }
        return {
            "verdict": verdict,
            "trace": _trace(state, "grade_context", started, verdict=verdict),
        }

    async def rewrite_query(self, state: AgentState) -> dict[str, Any]:
        """Rewrite the query after a rejected retrieval and bump the counter."""
        started = time.perf_counter()
        iteration = state["iteration"] + 1
        query = state["current_query"]
        source = "unchanged"
        prompt = (
            f"QUESTION: {state['question']}\n"
            f"PREVIOUS QUERY: {state['current_query']}\n"
            f"WHY IT FAILED: {state['verdict'].get('reason', '')}"
        )
        try:
            obj = extract_json_object(
                await self._llm.complete(REWRITE_SYSTEM, prompt, json_mode=True)
            )
            candidate = obj.get("query")
            if not isinstance(candidate, str) or not candidate.strip():
                raise ValueError("empty rewrite")
            query, source = candidate.strip(), "llm"
        except (LLMError, ValueError):
            pass
        return {
            "current_query": query,
            "iteration": iteration,
            "trace": _trace(
                state,
                "rewrite_query",
                started,
                iteration=iteration,
                query=query,
                source=source,
            ),
        }

    async def switch_retriever(self, state: AgentState) -> dict[str, Any]:
        """Fall back (once) to the retriever that has not been tried yet."""
        started = time.perf_counter()
        new_route = fallback_route(state["route"])
        if new_route is None:  # unreachable via decide_after_grade; defensive
            raise RuntimeError("no fallback retriever available")
        return {
            "route": new_route,
            "fallback_used": True,
            "tried_routes": [*state["tried_routes"], new_route],
            "current_query": state["question"],
            "trace": _trace(
                state,
                "switch_retriever",
                started,
                from_route=state["route"],
                to_route=new_route,
            ),
        }

    async def generate_answer(self, state: AgentState) -> dict[str, Any]:
        """Generate a cited answer from the accepted context."""
        started = time.perf_counter()
        items = state["items"]
        prompt = f"QUESTION: {state['question']}\n\nCONTEXT:\n{render_context(items)}"
        raw = await self._llm.complete(GENERATE_SYSTEM, prompt, json_mode=True)
        answerable: object = True
        citations_raw: object = []
        try:
            obj = extract_json_object(raw)
            answer = str(obj.get("answer", "")).strip()
            answerable = obj.get("answerable", True)
            citations_raw = obj.get("citations", [])
        except ValueError:
            answer = raw.strip()

        valid_ids = {i.chunk_id for i in items}
        citations: list[str] = []
        if isinstance(citations_raw, list):
            for cid in citations_raw:
                if isinstance(cid, str) and cid in valid_ids and cid not in citations:
                    citations.append(cid)

        if answerable is False or not answer:
            return {
                "answer": INSUFFICIENT_MESSAGE,
                "citations": [],
                "grounded": False,
                "insufficient": True,
                "trace": _trace(
                    state, "generate_answer", started, answerable=False, citations=0
                ),
            }
        return {
            "answer": answer,
            "citations": citations,
            "grounded": bool(citations),
            "insufficient": False,
            "trace": _trace(
                state,
                "generate_answer",
                started,
                answerable=True,
                citations=len(citations),
            ),
        }

    async def insufficient_evidence(self, state: AgentState) -> dict[str, Any]:
        """Terminal node: honest 'not enough evidence' answer."""
        started = time.perf_counter()
        return {
            "answer": INSUFFICIENT_MESSAGE,
            "citations": [],
            "grounded": False,
            "insufficient": True,
            "trace": _trace(
                state,
                "insufficient_evidence",
                started,
                tried_routes=state["tried_routes"],
            ),
        }
