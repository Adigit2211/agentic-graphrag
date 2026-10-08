"""Character-based text chunking with overlap."""

from __future__ import annotations


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split ``text`` into chunks of at most ``size`` characters.

    Chunks break on whitespace where possible and overlap by roughly
    ``overlap`` characters.

    Raises:
        ValueError: if ``size`` <= 0, ``overlap`` < 0 or ``overlap`` >= ``size``.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    text = text.strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            split = text.rfind(" ", start + size // 2, end)
            if split != -1:
                end = split
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
