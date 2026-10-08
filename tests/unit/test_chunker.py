import pytest

from ingestion.chunker import chunk_text


def test_short_text_is_single_chunk() -> None:
    assert chunk_text("hello world", size=100, overlap=10) == ["hello world"]


def test_chunks_respect_size_and_cover_text() -> None:
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    assert chunks[0].startswith("word0")
    assert chunks[-1].endswith("word199")


def test_consecutive_chunks_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_text(text, size=60, overlap=20)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert set(prev.split()) & set(nxt.split())


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_text("   \n ", size=50, overlap=5) == []


def test_chunking_is_deterministic() -> None:
    text = "alpha beta gamma delta " * 50
    assert chunk_text(text, 80, 10) == chunk_text(text, 80, 10)


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1), (10, 11)])
def test_invalid_parameters_raise(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("text", size, overlap)
