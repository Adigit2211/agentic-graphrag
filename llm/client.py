"""Async OpenAI-compatible chat client (Ollama by default, vLLM via config)."""

from __future__ import annotations

from typing import Protocol

import httpx


class LLMError(RuntimeError):
    """Raised when the LLM backend cannot produce a usable reply."""


class LLMClient(Protocol):
    """Minimal interface the rest of the codebase depends on."""

    async def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        """Return the assistant message for a system+user prompt."""
        ...

    async def ping(self) -> bool:
        """Return True if the backend is reachable."""
        ...

    async def aclose(self) -> None:
        """Release network resources."""
        ...


class OpenAICompatClient:
    """Talks to ``{base_url}/chat/completions`` of any OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "ollama",
        timeout_s: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
            transport=transport,
        )

    async def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        """Send one chat completion request (temperature 0)."""
        payload: dict[str, object] = {
            "model": self._model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = await self._client.post("chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"])
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

    async def ping(self) -> bool:
        """Check the ``/models`` endpoint."""
        try:
            resp = await self._client.get("models")
        except httpx.HTTPError:
            return False
        return resp.status_code == 200

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
