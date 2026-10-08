import pytest

from agent.grading import Decision, decide_after_grade, fallback_route, lexical_support
from agent.router_rules import parse_route, rule_route
from agent.state import AgentState, initial_state
from retrieval.models import RetrievedItem


def _item(text: str) -> RetrievedItem:
    return RetrievedItem(
        chunk_id="c", doc_id="d", text=text, score=1.0, origin="vector"
    )


@pytest.mark.parametrize("route", ["graph", "vector", "both"])
def test_parse_route_accepts_valid(route: str) -> None:
    assert parse_route(f'{{"route": "{route}"}}') == route


@pytest.mark.parametrize(
    "raw", ["nonsense", '{"route": "magic"}', '{"other": 1}', '{"route": null}']
)
def test_parse_route_rejects_invalid(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_route(raw)


def test_rule_route() -> None:
    assert rule_route("Who founded the company that acquired Helix Robotics?") == "both"
    assert rule_route("what is the capital city of the country?") == "vector"
    assert rule_route("Who is the tallest?") == "vector"  # question word + no entity


def test_lexical_support() -> None:
    items = [_item("Aurora Labs was founded by Mira Chen in Lisbon")]
    assert lexical_support("Who founded Aurora Labs?", items) == 1.0
    assert lexical_support("quantum entanglement experiments", items) == 0.0
    assert lexical_support("anything", []) == 0.0


def test_fallback_route() -> None:
    assert fallback_route("graph") == "vector"
    assert fallback_route("vector") == "graph"
    assert fallback_route("both") is None


def _state(**kw: object) -> AgentState:
    state = initial_state("q", top_k=3, max_iterations=2)
    state.update(kw)  # type: ignore[typeddict-item]
    return state


def test_decide_generate_when_relevant() -> None:
    assert decide_after_grade(_state(verdict={"relevant": True})) == "generate"


@pytest.mark.parametrize("route", ["graph", "vector", "both"])
@pytest.mark.parametrize("max_iterations", [0, 1, 2, 5])
def test_grader_loop_always_terminates(route: str, max_iterations: int) -> None:
    """Simulate a grader that always rejects; the loop must stop."""
    state = _state(
        route=route, max_iterations=max_iterations, verdict={"relevant": False}
    )
    retrievals = 1
    for _ in range(50):
        decision: Decision = decide_after_grade(state)
        if decision == "insufficient":
            break
        if decision == "rewrite":
            state["iteration"] += 1
        else:
            assert decision == "fallback"
            new_route = fallback_route(state["route"])
            assert new_route is not None
            state["route"] = new_route
            state["fallback_used"] = True
        retrievals += 1
    else:
        pytest.fail("grader loop did not terminate")
    expected_fallback = 0 if route == "both" else 1
    assert retrievals == 1 + max_iterations + expected_fallback
