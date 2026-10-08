import json

import httpx
import pytest

from llm.client import LLMError, OpenAICompatClient


def _client(handler: httpx.MockTransport) -> OpenAICompatClient:
    return OpenAICompatClient(
        base_url="http://llm.test/v1", model="m", api_key="k", transport=handler
    )


async def test_complete_sends_openai_payload_and_parses_reply() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    client = _client(httpx.MockTransport(handler))
    assert await client.complete("sys", "usr", json_mode=True) == "hi"
    assert seen["url"] == "http://llm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["model"] == "m"
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    await client.aclose()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, json={"unexpected": True}),
        httpx.Response(200, text="not json"),
    ],
)
async def test_complete_wraps_failures_in_llmerror(response: httpx.Response) -> None:
    client = _client(httpx.MockTransport(lambda request: response))
    with pytest.raises(LLMError):
        await client.complete("s", "u")
    await client.aclose()


async def test_ping() -> None:
    ok = _client(httpx.MockTransport(lambda r: httpx.Response(200, json={"data": []})))
    bad = _client(httpx.MockTransport(lambda r: httpx.Response(503)))
    assert await ok.ping() is True
    assert await bad.ping() is False

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    down = _client(httpx.MockTransport(refuse))
    assert await down.ping() is False
