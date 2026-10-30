"""
The ollama prefix refuses a base URL ending in /v1, Ollama's
OpenAI-compatible path, before any request reaches the backend (issue #10).
"""

import httpx
import pytest

from model_calling.errors import BadRequestError
from model_calling.library import Client

MESSAGES = [{"role": "user", "content": "hi"}]
EMBED_BODY = {"model": "m", "embeddings": [[0.1]]}
CHAT_BODY = {"model": "m", "message": {"role": "assistant", "content": "ok"}, "done": True}


async def chat(client: Client, **kwargs):
    return await client.acompletion("ollama/m", MESSAGES, **kwargs)


async def stream(client: Client, **kwargs):
    return [chunk async for chunk in await chat(client, stream=True, **kwargs)]


async def embed(client: Client, **kwargs):
    return await client.aembedding("ollama/m", ["hi"], **kwargs)


CALLS = {"chat": chat, "stream": stream, "embed": embed}


def recording_transport(sent: list) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url))
        if request.url.path.endswith("/api/embed"):
            return httpx.Response(200, json=EMBED_BODY)
        return httpx.Response(200, json=CHAT_BODY)

    return httpx.MockTransport(handler)


@pytest.mark.parametrize("call", sorted(CALLS))
@pytest.mark.parametrize("url", ["http://localhost:11434/v1", "http://localhost:11434/v1/"])
@pytest.mark.parametrize("where", ["config", "api_base"])
async def test_a_v1_base_url_is_refused_before_any_request(call, url, where):
    sent = []
    config = {"base_url": url if where == "config" else "http://localhost:11434"}
    kwargs = {"api_base": url} if where == "api_base" else {}
    async with Client(config={"ollama": {**config, "transport": recording_transport(sent)}}) as c:
        with pytest.raises(BadRequestError) as err:
            await CALLS[call](c, **kwargs)
    assert sent == []
    message = err.value.message
    assert err.value.provider == "ollama"
    assert "http://localhost:11434/v1 ends in /v1" in message, message
    assert "server root" in message and "openai/ prefix" in message, message


@pytest.mark.parametrize("call", sorted(CALLS))
@pytest.mark.parametrize(
    "url",
    ["http://localhost:11434", "http://localhost:11434/", "http://proxy.test/ollama/v1beta"],
)
async def test_other_base_urls_reach_the_backend(call, url):
    sent = []
    async with Client(config={"ollama": {"transport": recording_transport(sent)}}) as c:
        await CALLS[call](c, api_base=url)
    assert len(sent) == 1 and sent[0].startswith(url.rstrip("/") + "/api/"), sent
