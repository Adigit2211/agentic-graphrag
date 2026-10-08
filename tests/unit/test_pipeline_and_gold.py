"""Ingestion idempotency, failure handling and the gold-set retrieval regression."""

import json

import pytest

from app.container import Container
from ingestion.pipeline import IngestError, ingest_document
from tests.conftest import SAMPLE_DIR, FakeLLM, ingest_sample_docs, make_container


async def test_reingesting_does_not_duplicate_data(fake_llm: FakeLLM) -> None:
    container = make_container(fake_llm)
    await ingest_sample_docs(container)
    first_graph = container.graph_store.stats()
    first_vectors = len(container.vector_store)

    await ingest_sample_docs(container)

    assert container.graph_store.stats() == first_graph
    assert len(container.vector_store) == first_vectors
    assert first_graph["chunks"] == 3
    # Aurora Labs, Mira Chen, Lisbon, Helix Robotics, Porto, University of Coimbra
    assert first_graph["entities"] == 6


async def test_graph_provenance_links_edges_to_chunks(loaded: Container) -> None:
    seeds = loaded.graph_store.link_entities("aurora labs")
    reachable = loaded.graph_store.expand(seeds, hops=3, limit=100)
    ctx = loaded.graph_store.subgraph(reachable)
    assert ctx.edges, "expected edges around Aurora Labs"
    assert all(edge.chunk_ids for edge in ctx.edges)


async def test_ingest_fails_loudly_when_extraction_fails_everywhere(
    fake_llm: FakeLLM,
) -> None:
    fake_llm.extract_raw = "garbage"
    container = make_container(fake_llm)
    s = container.settings
    with pytest.raises(IngestError):
        await ingest_document(
            doc_id="x.md",
            text="Some text about nothing in particular.",
            llm=fake_llm,
            graph_store=container.graph_store,
            vector_store=container.vector_store,
            embedder=container.embedder,
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
            max_retries=0,
        )
    # chunks were still embedded, so dense retrieval keeps working
    assert len(container.vector_store) == 1


GOLD = json.loads((SAMPLE_DIR / "gold_questions.json").read_text())


@pytest.mark.parametrize("case", GOLD, ids=[g["expected_doc"] for g in GOLD])
async def test_gold_questions_retrieve_expected_doc_in_top2(
    loaded: Container, case: dict[str, str]
) -> None:
    """Regression guard for pipeline wiring on a tiny fictional corpus.

    This is NOT a quality benchmark: the corpus has three documents.
    """
    result = await loaded.retriever.retrieve(case["question"], "both", k=2)
    assert case["expected_doc"] in [i.doc_id for i in result.items]


async def test_multi_hop_question_reaches_second_hop_via_graph(
    loaded: Container,
) -> None:
    result = await loaded.retriever.retrieve(
        "Who founded the company that acquired Helix Robotics?", "graph", k=5
    )
    docs = [i.doc_id for i in result.items]
    assert "aurora_labs.md" in docs
    assert all(i.origin == "graph" for i in result.items)
    names = {n.name for n in result.graph_context.nodes}
    assert {"Helix Robotics", "Aurora Labs", "Mira Chen"} <= names


async def test_origin_tags_distinguish_retrievers(loaded: Container) -> None:
    both = await loaded.retriever.retrieve("Helix Robotics acquired", "both", k=3)
    vector = await loaded.retriever.retrieve("Helix Robotics acquired", "vector", k=3)
    assert {i.origin for i in vector.items} == {"vector"}
    assert "both" in {i.origin for i in both.items}
