"""Grader fallbacks and the loop-control decision."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from agent.state import AgentState
from retrieval.models import Mode, RetrievedItem

Decision = Literal["generate", "rewrite", "fallback", "insufficient"]

_STOPWORDS = {
    "the", "and", "for", "that", "with", "was", "were", "who", "what", "when",
    "where", "which", "why", "how", "whose", "did", "does", "has", "have", "had",
    "are", "is", "of", "in", "on", "at", "to", "by", "from", "this", "these",
    "those", "their", "its", "his", "her", "than", "then", "there", "into",
}  # fmt: skip


def lexical_support(question: str, items: Sequence[RetrievedItem]) -> float:
    """Fraction of the question's content words that appear in the context.

    Used only when the LLM grader's output is unusable. It is a crude proxy
    for relevance, not a groundedness check.
    """
    wanted = {
        t
        for t in re.findall(r"\w+", question.lower())
        if len(t) >= 3 and t not in _STOPWORDS
    }
    if not wanted or not items:
        return 0.0
    context = set(re.findall(r"\w+", " ".join(i.text for i in items).lower()))
    return len(wanted & context) / len(wanted)


def fallback_route(route: str) -> Mode | None:
    """The 'other' retriever to try once, or None if both were already used."""
    if route == "graph":
        return "vector"
    if route == "vector":
        return "graph"
    return None


def decide_after_grade(state: AgentState) -> Decision:
    """Choose the next step after grading.

    Termination argument: ``rewrite`` increments ``iteration`` and is only
    chosen while ``iteration < max_iterations``; ``fallback`` is chosen at most
    once (guarded by ``fallback_used``) and never resets ``iteration``. So the
    loop performs at most ``max_iterations + 2`` retrievals.
    """
    if state["verdict"].get("relevant") is True:
        return "generate"
    if state["iteration"] < state["max_iterations"]:
        return "rewrite"
    if not state["fallback_used"] and fallback_route(state["route"]) is not None:
        return "fallback"
    return "insufficient"
