"""Shared fixtures: a scripted fake LLM and a pre-loaded container."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest_asyncio

from app.config import Settings
from app.container import Container, build_container
from ingestion.loaders import decode_document, doc_id_from_filename
from ingestion.pipeline import ingest_document
from retrieval.embeddings import HashingEmbedder

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"

EXTRACTION_MAP: dict[str, dict[str, Any]] = {
    "founded by Mira Chen": {
        "entities": [
            {"name": "Aurora Labs", "type": "ORG"},
            {"name": "Mira Chen", "type": "PERSON"},
            {"name": "Lisbon", "type": "LOCATION"},
        ],
        "relations": [
            {
                "subject": "Aurora Labs",
                "predicate": "founded by",
                "object": "Mira Chen",
            },
            {
                "subject": "Aurora Labs",
                "predicate": "headquartered in",
                "object": "Lisbon",
            },
        ],
    },
    "warehouse drones": {
        "entities": [
            {"name": "Helix Robotics", "type": "ORG"},
            {"name": "Aurora Labs", "type": "ORG"},
            {"name": "Porto", "type": "LOCATION"},
        ],
        "relations": [
            {
                "subject": "Helix Robotics",
                "predicate": "acquired by",
                "object": "Aurora Labs",
            },
            {
                "subject": "Helix Robotics",
                "predicate": "operates from",
                "object": "Porto",
            },
        ],
    },
    "University of Coimbra": {
        "entities": [
            {"name": "Mira Chen", "type": "PERSON"},
            {"name": "University of Coimbra", "type": "ORG"},
            {"name": "Aurora Labs", "type": "ORG"},
        ],
        "relations": [
            {
                "subject": "Mira Chen",
                "predicate": "studied at",
                "object": "University of Coimbra",
            },
            {
                "subject": "Mira Chen",
                "predicate": "chief executive of",
                "object": "Aurora Labs",
            },
        ],
    },
}


class FakeLLM:
    """Scripted LLM. Replies are chosen from the ``[task:...]`` tag in the prompt."""

    def __init__(self) -> None:
        self.route_raw = '{"route": "both"}'
        self.grade_raw = '{"relevant": true, "reason": "ok"}'
        self.rewrite_raw = '{"query": "rewritten query"}'
        self.generate_raw: str | None = None
        self.extract_raw: str | None = None
        self.extraction_map = EXTRACTION_MAP
        self.calls: list[str] = []
        self.healthy = True

    async def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        match = re.match(r"\[task:(\w+)\]", system)
        assert match, f"prompt has no task tag: {system[:40]!r}"
        task = match.group(1)
        self.calls.append(task)
        if task == "route":
            return self.route_raw
        if task == "grade":
            return self.grade_raw
        if task == "rewrite":
            return self.rewrite_raw
        if task == "generate":
            if self.generate_raw is not None:
                return self.generate_raw
            ids = re.findall(r"^\[([0-9a-f]{16})\]", user, re.MULTILINE)
            return json.dumps(
                {"answerable": True, "answer": "fake answer", "citations": ids[:1]}
            )
        if task == "extract":
            if self.extract_raw is not None:
                return self.extract_raw
            for key, payload in self.extraction_map.items():
                if key in user:
                    return json.dumps(payload)
            return '{"entities": [], "relations": []}'
        raise AssertionError(f"unknown task {task}")

    async def ping(self) -> bool:
        return self.healthy

    async def aclose(self) -> None:
        return None


def make_container(llm: FakeLLM) -> Container:
    """Container with the fake LLM and the offline hashing embedder."""
    settings = Settings(embedder="hashing", top_k=3, max_iterations=2)
    return build_container(settings, llm=llm, embedder=HashingEmbedder())


async def ingest_sample_docs(container: Container) -> None:
    """Ingest every file in data/sample/*.md through the real pipeline."""
    s = container.settings
    for path in sorted(SAMPLE_DIR.glob("*.md")):
        await ingest_document(
            doc_id=doc_id_from_filename(path.name),
            text=decode_document(path.name, path.read_bytes()),
            llm=container.llm,
            graph_store=container.graph_store,
            vector_store=container.vector_store,
            embedder=container.embedder,
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
            max_retries=s.extraction_max_retries,
        )


@pytest_asyncio.fixture
async def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest_asyncio.fixture
async def loaded(fake_llm: FakeLLM) -> Container:
    """A container with the sample corpus already ingested."""
    container = make_container(fake_llm)
    await ingest_sample_docs(container)
    fake_llm.calls.clear()
    return container
