"""HTTP-level tests using httpx's ASGI transport."""

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest_asyncio

from app.container import Container
from app.main import create_app
from tests.conftest import SAMPLE_DIR, FakeLLM, make_container


@pytest_asyncio.fixture
async def client(loaded: Container) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(loaded)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health_reports_components_separately(client: httpx.AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["llm"]["ok"] is True
    assert body["graph_store"]["chunks"] == 3
    assert body["vector_store"]["vectors"] == 3


async def test_health_is_503_when_llm_down(
    client: httpx.AsyncClient, fake_llm: FakeLLM
) -> None:
    fake_llm.healthy = False
    resp = await client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["llm"]["ok"] is False
    assert resp.json()["graph_store"]["ok"] is True


async def test_query_returns_answer_citations_context_and_trace(
    client: httpx.AsyncClient,
) -> None:
    resp = await client.post(
        "/query",
        json={
            "query": "Who founded the company that acquired Helix Robotics?",
            "options": {"top_k": 3},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "fake answer"
    assert body["grounded"] is True
    assert body["insufficient_evidence"] is False
    assert body["citations"][0]["doc_id"].endswith(".md")
    assert {n["name"] for n in body["graph_context"]["nodes"]} >= {"Helix Robotics"}
    assert {i["origin"] for i in body["retrieved"]} <= {"graph", "vector", "both"}
    assert [e["node"] for e in body["trace"]][0] == "route_query"


async def test_query_validation(client: httpx.AsyncClient) -> None:
    assert (await client.post("/query", json={"query": ""})).status_code == 422
    assert (await client.post("/query", json={})).status_code == 422
    bad = {"query": "x", "options": {"top_k": 0}}
    assert (await client.post("/query", json=bad)).status_code == 422


async def test_query_reports_llm_failure_as_502(
    client: httpx.AsyncClient, fake_llm: FakeLLM
) -> None:
    from llm.client import LLMError

    async def boom(system: str, user: str, *, json_mode: bool = False) -> str:
        raise LLMError("backend down")

    fake_llm.complete = boom  # type: ignore[method-assign]
    resp = await client.post("/query", json={"query": "Who is Mira Chen?"})
    # route/grade degrade gracefully to rules; the generator error surfaces as 502
    assert resp.status_code == 502


async def test_ingest_roundtrip_then_query(fake_llm: FakeLLM) -> None:
    container = make_container(fake_llm)
    app = create_app(container)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        path = SAMPLE_DIR / "aurora_labs.md"
        resp = await c.post(
            "/ingest", files={"file": (path.name, path.read_bytes(), "text/markdown")}
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        status = {}
        for _ in range(100):
            status = (await c.get(f"/ingest/{job_id}")).json()
            if status["status"] in ("done", "failed"):
                break
            await asyncio.sleep(0.02)
        assert status["status"] == "done", status
        assert status["chunks"] == 1
        assert status["entities"] == 3
        assert status["relations"] == 2

        answer = await c.post("/query", json={"query": "Who founded Aurora Labs?"})
        assert answer.status_code == 200
        assert answer.json()["citations"][0]["doc_id"] == "aurora_labs.md"


async def test_ingest_rejects_unsupported_and_empty_files(
    client: httpx.AsyncClient,
) -> None:
    pdf = await client.post(
        "/ingest", files={"file": ("paper.pdf", b"%PDF-1.4", "application/pdf")}
    )
    assert pdf.status_code == 415
    empty = await client.post(
        "/ingest", files={"file": ("a.txt", b"  \n", "text/plain")}
    )
    assert empty.status_code == 422


async def test_unknown_job_is_404(client: httpx.AsyncClient) -> None:
    assert (await client.get("/ingest/nope")).status_code == 404


async def test_failed_ingest_job_reports_error(fake_llm: FakeLLM) -> None:
    fake_llm.extract_raw = "garbage"
    container = make_container(fake_llm)
    app = create_app(container)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            "/ingest", files={"file": ("n.txt", b"Plain text.", "text/plain")}
        )
        job_id = resp.json()["job_id"]
        status = {}
        for _ in range(100):
            status = (await c.get(f"/ingest/{job_id}")).json()
            if status["status"] in ("done", "failed"):
                break
            await asyncio.sleep(0.02)
    assert status["status"] == "failed"
    assert "extraction failed" in status["error"]
