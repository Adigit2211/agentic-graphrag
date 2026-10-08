"""In-memory ingest job tracking (lost on restart; see README Limitations)."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel


class JobStatus(BaseModel):
    """State of one background ingest job."""

    job_id: str
    filename: str
    status: Literal["queued", "running", "done", "failed"] = "queued"
    doc_id: str | None = None
    chunks: int | None = None
    entities: int | None = None
    relations: int | None = None
    failed_chunks: int | None = None
    error: str | None = None


class JobStore:
    """Process-local registry of ingest jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobStatus] = {}

    def create(self, filename: str) -> JobStatus:
        """Register a new queued job."""
        job = JobStatus(job_id=uuid.uuid4().hex, filename=filename)
        self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> JobStatus | None:
        """Return a job by ID, or None."""
        return self._jobs.get(job_id)

    def update(self, job_id: str, **changes: object) -> JobStatus:
        """Apply field updates to a job and return the new state."""
        job = self._jobs[job_id].model_copy(update=changes)
        self._jobs[job_id] = job
        return job
