"""LangGraph wiring for the agentic retrieval loop."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.grading import decide_after_grade
from agent.nodes import AgentNodes
from agent.state import AgentState, initial_state
from llm.client import LLMClient
from retrieval.hybrid import HybridRetriever


def build_workflow(nodes: AgentNodes) -> Any:
    """Compile the state machine.

    route_query -> retrieve_context -> grade_context -> one of:
      generate_answer (relevant) | rewrite_query -> retrieve_context |
      switch_retriever -> retrieve_context | insufficient_evidence

    Node names deliberately differ from state keys (LangGraph forbids reuse).
    """
    graph = StateGraph(AgentState)
    graph.add_node("route_query", nodes.route_query)
    graph.add_node("retrieve_context", nodes.retrieve_context)
    graph.add_node("grade_context", nodes.grade_context)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("switch_retriever", nodes.switch_retriever)
    graph.add_node("generate_answer", nodes.generate_answer)
    graph.add_node("insufficient_evidence", nodes.insufficient_evidence)

    graph.add_edge(START, "route_query")
    graph.add_edge("route_query", "retrieve_context")
    graph.add_edge("retrieve_context", "grade_context")
    graph.add_conditional_edges(
        "grade_context",
        decide_after_grade,
        {
            "generate": "generate_answer",
            "rewrite": "rewrite_query",
            "fallback": "switch_retriever",
            "insufficient": "insufficient_evidence",
        },
    )
    graph.add_edge("rewrite_query", "retrieve_context")
    graph.add_edge("switch_retriever", "retrieve_context")
    graph.add_edge("generate_answer", END)
    graph.add_edge("insufficient_evidence", END)
    return graph.compile()


class GraphRagAgent:
    """Facade: run the compiled workflow for one question."""

    def __init__(self, llm: LLMClient, retriever: HybridRetriever) -> None:
        self._workflow = build_workflow(AgentNodes(llm, retriever))

    async def answer(
        self, question: str, *, top_k: int, max_iterations: int
    ) -> AgentState:
        """Run the agent and return the final state (including the trace)."""
        final: AgentState = await self._workflow.ainvoke(
            initial_state(question, top_k, max_iterations)
        )
        return final
