"""Helpers for pulling JSON out of free-form LLM replies."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(raw: str) -> dict[str, Any]:
    """Parse the first JSON object found in ``raw``.

    Tolerates markdown code fences and leading/trailing prose.

    Raises:
        ValueError: if no valid JSON object can be found.
    """
    text = raw.strip()
    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in LLM reply")
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in LLM reply: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("LLM reply JSON is not an object")
    return obj
