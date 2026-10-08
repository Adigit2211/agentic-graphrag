"""Reciprocal Rank Fusion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse several ranked lists of IDs.

    Each item scores ``sum(1 / (k + rank))`` over the lists containing it,
    with ranks starting at 1. Duplicates within one list count once (first
    occurrence). Ties are broken by ID so results are deterministic.

    Args:
        rankings: ``source_name -> ranked IDs`` (best first).
        k: RRF damping constant (60 in the original paper).
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in rankings.values():
        seen: set[str] = set()
        rank = 0
        for item in ranked:
            if item in seen:
                continue
            seen.add(item)
            rank += 1
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
