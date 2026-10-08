"""Router helpers: LLM-reply parsing and the rule-based fallback."""

from __future__ import annotations

import re

from llm.json_utils import extract_json_object
from retrieval.models import Mode

ROUTES: tuple[Mode, ...] = ("graph", "vector", "both")

_QUESTION_WORDS = {"who", "what", "when", "where", "which", "why", "how", "whose"}


def as_mode(value: object) -> Mode:
    """Validate ``value`` as a route name.

    Raises:
        ValueError: if ``value`` is not one of ``graph``, ``vector``, ``both``.
    """
    for route in ROUTES:
        if value == route:
            return route
    raise ValueError(f"invalid route: {value!r}")


def parse_route(raw: str) -> Mode:
    """Parse ``{"route": ...}`` from an LLM reply.

    Raises:
        ValueError: if the reply is not JSON or names an unknown route.
    """
    return as_mode(extract_json_object(raw).get("route"))


def rule_route(question: str) -> Mode:
    """Deterministic fallback when the LLM router output is unusable.

    Questions that mention at least one proper-noun-like word (capitalized and
    not the first word or a question word) are routed to ``both``; the rest go
    to ``vector``. ``both`` is the safest default because it covers the graph
    path without giving up dense recall.
    """
    words = re.findall(r"[A-Za-z][\w'-]*", question)
    proper = [
        w
        for i, w in enumerate(words)
        if w[0].isupper() and i > 0 and w.lower() not in _QUESTION_WORDS
    ]
    return "both" if proper else "vector"
