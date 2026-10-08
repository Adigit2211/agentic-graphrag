"""Turn uploaded files into plain text."""

from __future__ import annotations

import re
from pathlib import Path

SUPPORTED_SUFFIXES = {".txt", ".md"}


class UnsupportedFormatError(ValueError):
    """Raised for file types v0.1 cannot read (e.g. PDF)."""


def doc_id_from_filename(filename: str) -> str:
    """Derive a stable, filesystem-safe document ID from an upload name."""
    name = Path(filename).name
    return re.sub(r"[^\w.\-]+", "_", name) or "document"


def decode_document(filename: str, data: bytes) -> str:
    """Decode an uploaded ``.txt`` / ``.md`` file.

    Raises:
        UnsupportedFormatError: for any other extension.
        ValueError: if the file is empty after stripping whitespace.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFormatError(
            f"unsupported file type {suffix or '(none)'}; "
            f"supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
    text = data.decode("utf-8", errors="replace").strip()
    if not text:
        raise ValueError("document is empty")
    return text
