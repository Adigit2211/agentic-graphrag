"""Name normalization and deterministic IDs (the basis of idempotent ingestion)."""

from __future__ import annotations

import hashlib
import re
import unicodedata


def normalize_name(name: str) -> str:
    """Lower-case, NFKC-normalize and reduce to space-separated word tokens.

    The same function is applied to entity names and to user queries, so entity
    linking can be a whole-phrase match on the normalized strings.
    """
    text = unicodedata.normalize("NFKC", name).lower()
    return " ".join(re.findall(r"\w+", text))


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def entity_id(norm_name: str) -> str:
    """Deterministic entity ID derived from the normalized name."""
    return _digest("entity", norm_name)


def chunk_id(doc_id: str, idx: int, text: str) -> str:
    """Deterministic chunk ID; identical input always yields the same ID."""
    return _digest("chunk", doc_id, str(idx), text)
