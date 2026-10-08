"""``POST /ingest`` and ``GET /ingest/{job_id}``."""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile

from app.api.deps import ContainerDep
from app.container import Container
from app.schemas import IngestAccepted
from ingestion.jobs import JobStatus
from ingestion.loaders import (
    UnsupportedFormatError,
    decode_document,
    doc_id_from_filename,
)
from ingestion.pipeline import IngestError, ingest_document

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingest"])


async def _run_job(container: Container, job_id: str, doc_id: str, text: str) -> None:
    """Background task: ingest one document and record the outcome."""
    container.jobs.update(job_id, status="running", doc_id=doc_id)
    s = container.settings
    try:
        report = await ingest_document(
            doc_id=doc_id,
            text=text,
            llm=container.llm,
            graph_store=container.graph_store,
            vector_store=container.vector_store,
            embedder=container.embedder,
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
            max_retries=s.extraction_max_retries,
        )
    except IngestError as exc:
        container.jobs.update(job_id, status="failed", error=str(exc))
        logger.warning("ingest job %s failed: %s", job_id, exc)
        return
    except Exception as exc:  # last-resort guard so the job never stays 'running'
        container.jobs.update(job_id, status="failed", error=f"internal error: {exc}")
        logger.exception("ingest job %s crashed", job_id)
        return
    container.jobs.update(
        job_id,
        status="done",
        chunks=report.chunks,
        entities=report.entities,
        relations=report.relations,
        failed_chunks=report.failed_chunks,
    )


@router.post("/ingest", status_code=202, response_model=IngestAccepted)
async def ingest(
    file: UploadFile, background: BackgroundTasks, container: ContainerDep
) -> IngestAccepted:
    """Accept a ``.txt`` / ``.md`` upload and process it in the background."""
    data = await file.read()
    if len(data) > container.settings.max_upload_bytes:
        raise HTTPException(413, "file too large")
    filename = file.filename or "upload.txt"
    try:
        text = decode_document(filename, data)
    except UnsupportedFormatError as exc:
        raise HTTPException(415, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    job = container.jobs.create(filename)
    background.add_task(
        _run_job, container, job.job_id, doc_id_from_filename(filename), text
    )
    return IngestAccepted(job_id=job.job_id, status=job.status)


@router.get("/ingest/{job_id}", response_model=JobStatus)
async def ingest_status(job_id: str, container: ContainerDep) -> JobStatus:
    """Return the state of an ingest job."""
    job = container.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job id")
    return job
