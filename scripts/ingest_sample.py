"""Upload ``data/sample/*.md`` to a running API and wait for each ingest job."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import httpx


def _wait(client: httpx.Client, job_id: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        status: dict[str, Any] = client.get(f"/ingest/{job_id}").json()
        if status["status"] in ("done", "failed"):
            return status
        if time.monotonic() > deadline:
            return {"status": "failed", "error": "timed out waiting for job"}
        time.sleep(1.0)


def main() -> int:
    """Entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--dir", type=Path, default=Path("data/sample"))
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    files = sorted(args.dir.glob("*.md"))
    if not files:
        print(f"no .md files found in {args.dir}", file=sys.stderr)
        return 1

    failures = 0
    try:
        with httpx.Client(base_url=args.url, timeout=30.0) as client:
            for path in files:
                resp = client.post(
                    "/ingest",
                    files={"file": (path.name, path.read_bytes(), "text/markdown")},
                )
                resp.raise_for_status()
                status = _wait(client, resp.json()["job_id"], args.timeout)
                detail = (
                    f"{status.get('chunks')} chunks, {status.get('entities')} "
                    f"entities, {status.get('relations')} relations"
                    if status["status"] == "done"
                    else status.get("error")
                )
                print(f"{path.name}: {status['status']} ({detail})")
                failures += status["status"] != "done"
    except httpx.ConnectError:
        print(
            f"cannot reach {args.url}; start the API with `make run`", file=sys.stderr
        )
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
