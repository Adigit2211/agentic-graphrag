"""End-to-end LangGraph runs against a scripted LLM."""

from agent.state import AgentState
from app.container import Container
from tests.conftest import FakeLLM

QUESTION = "Who founded the company that acquired Helix Robotics?"


def _nodes(state: AgentState) -> list[str]:
    return [event["node"] for event in state["trace"]]


async def test_happy_path_visits_expected_nodes(loaded: Container) -> None:
    state = await loaded.agent.answer(QUESTION, top_k=3, max_iterations=2)
    assert _nodes(state) == [
        "route_query",
        "retrieve_context",
        "grade_context",
        "generate_answer",
    ]
    assert state["answer"] == "fake answer"
    assert state["grounded"] is True
    assert state["insufficient"] is False
    assert state["citations"] and state["citations"][0] in {
        i.chunk_id for i in state["items"]
    }
    assert all("latency_ms" in event for event in state["trace"])


async def test_rejecting_grader_rewrites_falls_back_then_gives_up(
    loaded: Container, fake_llm: FakeLLM
) -> None:
    fake_llm.route_raw = '{"route": "graph"}'
    fake_llm.grade_raw = '{"relevant": false, "reason": "missing evidence"}'
    state = await loaded.agent.answer(QUESTION, top_k=3, max_iterations=2)
    nodes = _nodes(state)
    assert nodes.count("retrieve_context") == 4  # initial + 2 rewrites + 1 fallback
    assert nodes.count("rewrite_query") == 2
    assert nodes.count("switch_retriever") == 1
    assert nodes[-1] == "insufficient_evidence"
    assert state["insufficient"] is True
    assert state["citations"] == []
    assert state["tried_routes"] == ["graph", "vector"]
    assert "generate" not in fake_llm.calls


async def test_both_route_has_no_fallback(loaded: Container, fake_llm: FakeLLM) -> None:
    fake_llm.grade_raw = '{"relevant": false, "reason": "no"}'
    state = await loaded.agent.answer(QUESTION, top_k=3, max_iterations=1)
    nodes = _nodes(state)
    assert nodes.count("retrieve_context") == 2
    assert "switch_retriever" not in nodes
    assert state["insufficient"] is True


async def test_zero_iterations_skips_rewrite(
    loaded: Container, fake_llm: FakeLLM
) -> None:
    fake_llm.route_raw = '{"route": "vector"}'
    fake_llm.grade_raw = '{"relevant": false, "reason": "no"}'
    state = await loaded.agent.answer(QUESTION, top_k=3, max_iterations=0)
    nodes = _nodes(state)
    assert "rewrite_query" not in nodes
    assert nodes.count("switch_retriever") == 1


async def test_invalid_router_output_uses_rule_fallback(
    loaded: Container, fake_llm: FakeLLM
) -> None:
    fake_llm.route_raw = "I think you should use the graph, probably."
    state = await loaded.agent.answer(QUESTION, top_k=3, max_iterations=1)
    route_event = state["trace"][0]
    assert route_event["node"] == "route_query"
    assert route_event["source"] == "rules"
    assert route_event["route"] == "both"  # proper nouns present -> both
    assert state["insufficient"] is False


async def test_invalid_grader_output_uses_lexical_fallback(
    loaded: Container, fake_llm: FakeLLM
) -> None:
    fake_llm.grade_raw = "no idea"
    state = await loaded.agent.answer(QUESTION, top_k=3, max_iterations=0)
    grade_event = next(e for e in state["trace"] if e["node"] == "grade_context")
    assert grade_event["verdict"]["source"] == "rules"
    assert state["insufficient"] is False  # corpus covers every question word


async def test_unanswerable_generation_becomes_insufficient(
    loaded: Container, fake_llm: FakeLLM
) -> None:
    fake_llm.generate_raw = '{"answerable": false, "answer": "", "citations": []}'
    state = await loaded.agent.answer(QUESTION, top_k=3, max_iterations=0)
    assert state["insufficient"] is True
    assert state["grounded"] is False


async def test_hallucinated_citations_are_dropped(
    loaded: Container, fake_llm: FakeLLM
) -> None:
    fake_llm.generate_raw = (
        '{"answerable": true, "answer": "Mira Chen", "citations": ["deadbeefdeadbeef"]}'
    )
    state = await loaded.agent.answer(QUESTION, top_k=3, max_iterations=0)
    assert state["answer"] == "Mira Chen"
    assert state["citations"] == []
    assert state["grounded"] is False  # surfaced honestly rather than hidden
