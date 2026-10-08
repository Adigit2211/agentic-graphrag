"""Idempotent ingestion: chunk -> embed -> extract graph -> write stores."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from graph.models import ChunkRecord, EntityRecord
from graph.naming import chunk_id
from graph.store import GraphStore
from ingestion.chunker import chunk_text
from ingestion.extractor import ExtractionError, extract_graph
from ingestion.normalizer import normalize_extraction
from llm.client import LLMClient, LLMError
from retrieval.dense import InMemoryVectorStore
from retrieval.embeddings import Embedder

logger = logging.getLogger(__name__)


class IngestError(RuntimeError):
    """Raised when a document could not be ingested."""


@dataclass(frozen=True)
class IngestReport:
    """Summary of one ingestion run."""

    doc_id: str
    chunks: int
    entities: int
    relations: int
    failed_chunks: int


async def ingest_document(
    *,
    doc_id: str,
    text: str,
    llm: LLMClient,
    graph_store: GraphStore,
    vector_store: InMemoryVectorStore,
    embedder: Embedder,
    chunk_size: int,
    chunk_overlap: int,
    max_retries: int,
) -> IngestReport:
    """Ingest one document.

    Re-ingesting identical text is a no-op: chunk and entity IDs are
    deterministic and every store write is an upsert.

    Raises:
        IngestError: if the text yields no chunks, or extraction failed for
            every chunk (chunks are still embedded and searchable in that case).
    """
    pieces = chunk_text(text, chunk_size, chunk_overlap)
    if not pieces:
        raise IngestError("document produced no chunks")
    ids = [chunk_id(doc_id, i, piece) for i, piece in enumerate(pieces)]

    for i, (cid, piece) in enumerate(zip(ids, pieces, strict=True)):
        graph_store.add_chunk(ChunkRecord(id=cid, doc_id=doc_id, idx=i, text=piece))

    vectors = await asyncio.to_thread(embedder.embed, pieces)
    for cid, vec in zip(ids, vectors, strict=True):
        vector_store.upsert(cid, vec)

    entity_ids: set[str] = set()
    relation_keys: set[tuple[str, str, str]] = set()
    failed = 0
    for cid, piece in zip(ids, pieces, strict=True):
        try:
            extraction = await extract_graph(llm, piece, max_retries)
        except (ExtractionError, LLMError) as exc:
            failed += 1
            logger.warning("extraction failed for chunk %s: %s", cid, exc)
            continue
        entities, relations = normalize_extraction(extraction)
        for ent in entities:
            graph_store.add_entity(
                EntityRecord(
                    id=ent.id, name=ent.name, norm_name=ent.norm_name, type=ent.type
                )
            )
            graph_store.add_mention(ent.id, cid)
            entity_ids.add(ent.id)
        for rel in relations:
            graph_store.add_relation(rel.subject_id, rel.predicate, rel.object_id, cid)
            relation_keys.add((rel.subject_id, rel.predicate, rel.object_id))

    if failed == len(pieces):
        raise IngestError(
            f"graph extraction failed for all {failed} chunks "
            "(is the LLM backend running?); chunks were still embedded"
        )
    return IngestReport(
        doc_id=doc_id,
        chunks=len(pieces),
        entities=len(entity_ids),
        relations=len(relation_keys),
        failed_chunks=failed,
    )
