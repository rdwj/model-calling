"""
Shared provider HTTP behavior: error mapping, stream lifecycle, callbacks.

Exercised through OllamaProvider against an httpx.MockTransport.
"""

import json

import httpx
import pytest

from model_calling.callbacks import Callback
from model_calling.core import Message, MessageStart, Request, StreamError, Text
from model_calling.errors import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from model_calling.gateway import chat_completions, client_status
from model_calling.providers import HttpClients, OllamaProvider, provider_for

DONE_LINE = json.dumps(
    {"model": "m", "message": {"role": "assistant", "content": ""}, "done": True}
)
TEXT_LINE = json.dumps(
    {"model": "m", "message": {"role": "assistant", "content": "hi"}, "done": False}
)


class Recorder(Callback):
    def __init__(self, fail: bool = False):
        self.hooks = []
        self.fail = fail

    async def _record(self, hook):
        self.hooks.append(hook)
        if self.fail:
            raise RuntimeError(f"callback {hook} broke")

    async def on_request_start(self, model, messages, **kwargs):
        await self._record("start")

    async def on_request_end(self, model, messages, response, **kwargs):
        await self._record("end")

    async def on_stream_chunk(self, model, messages, chunk, **kwargs):
        await self._record("chunk")

    async def on_error(self, model, messages, error, **kwargs):
        await self._record("error")


@pytest.fixture
async def backend():
    """A shared client whose transport answers with the handler the test sets."""
    state = {}
    transport = httpx.MockTransport(lambda req: state["handler"](req))
    async with httpx.AsyncClient(transport=transport) as client:
        state["client"] = client
        yield state


def make_provider(backend, *callbacks):
    return OllamaProvider(
        model="m",
        base_url="http://ollama.test",
        callbacks=callbacks,
        http_client=backend["client"],
    )


def request(stream=False):
    return Request(
        model="ollama/m", messages=[Message(role="user", content=[Text(text="x")])], stream=stream
    )


def ndjson(*lines):
    return lambda req: httpx.Response(200, text="\n".join(lines) + "\n")


def broken_stream(req):
    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield (TEXT_LINE + "\n").encode()
            raise httpx.ReadError("connection reset")

    return httpx.Response(200, stream=Broken())


UNREADABLE_TOOL_CALL = {
    "model": "m",
    "message": {"tool_calls": [{"function": {"arguments": {}}}]},
    "done": True,
}


@pytest.mark.parametrize(
    "handler, error_type, status",
    [
        (
            lambda req: httpx.Response(200, text="<html>not json</html>"),
            APIResponseValidationError,
            200,
        ),
        (lambda req: httpx.Response(500, json={"error": "boom"}), InternalServerError, 500),
        (lambda req: httpx.Response(429, json={"error": "slow down"}), RateLimitError, 429),
        (lambda req: httpx.Response(401, json={"error": "bad key"}), AuthenticationError, 401),
        (lambda req: httpx.Response(403, json={"error": "forbidden"}), PermissionDeniedError, 403),
        (lambda req: httpx.Response(418, json={"error": "teapot"}), APIStatusError, 418),
        (
            lambda req: httpx.Response(200, json=UNREADABLE_TOOL_CALL),
            APIResponseValidationError,
            None,
        ),
        (
            lambda req: (_ for _ in ()).throw(httpx.ConnectError("refused")),
            APIConnectionError,
            None,
        ),
        (lambda req: (_ for _ in ()).throw(httpx.ReadTimeout("slow")), APITimeoutError, None),
    ],
    ids=[
        "non_json",
        "server_error",
        "rate_limited",
        "credential_refused",
        "credential_forbidden",
        "other_status",
        "unreadable_tool_call",
        "unreachable",
        "timeout",
    ],
)
async def test_complete_raises_the_backends_real_status(backend, handler, error_type, status):
    backend["handler"] = handler
    recorder = Recorder()
    with pytest.raises(error_type) as err:
        await make_provider(backend, recorder).complete(request())
    assert (err.value.status, err.value.provider) == (status, "ollama")
    assert recorder.hooks == ["start", "error"]


async def test_error_carries_the_raw_body(backend):
    backend["handler"] = lambda req: httpx.Response(404, json={"error": "model 'm' not found"})
    with pytest.raises(NotFoundError) as err:
        await make_provider(backend).complete(request())
    assert err.value.body == {"error": "model 'm' not found"}
    assert "model 'm' not found" in err.value.message


@pytest.mark.parametrize(
    "error, status",
    [
        (NotFoundError("x", status=404), 404),
        (RateLimitError("x", status=429), 429),
        (BadRequestError("x", status=400), 400),
        (AuthenticationError("x", status=401), 502),
        (PermissionDeniedError("x", status=403), 502),
        (InternalServerError("x", status=503), 502),
        (APIResponseValidationError("x", status=200), 502),
        (APIConnectionError("x"), 502),
        (APIError("x"), 502),
        (APITimeoutError("x"), 504),
    ],
    ids=lambda v: type(v).__name__ if isinstance(v, Exception) else str(v),
)
def test_gateway_client_status(error, status):
    """The gateway hides backend 5xx and credential refusals behind a 502."""
    assert client_status(error) == status


async def test_stream_mid_stream_transport_error_is_connection_error(backend):
    backend["handler"] = broken_stream
    recorder = Recorder()
    events = make_provider(backend, recorder).stream(request(stream=True))
    with pytest.raises(APIConnectionError, match="stream broke"):
        async for _ in events:
            pass
    assert recorder.hooks[-1] == "error"


async def test_stream_error_line_is_event_and_error_callback(backend):
    backend["handler"] = ndjson(TEXT_LINE, json.dumps({"error": "model crashed"}))
    recorder = Recorder()
    events = [e async for e in make_provider(backend, recorder).stream(request(stream=True))]
    assert isinstance(events[-1], StreamError)
    assert recorder.hooks == ["start", "chunk", "chunk", "error"]


async def test_consumer_closing_early_still_ends_callbacks(backend):
    backend["handler"] = ndjson(TEXT_LINE, TEXT_LINE, DONE_LINE)
    recorder = Recorder()
    events = make_provider(backend, recorder).stream(request(stream=True))
    assert isinstance(await anext(events), MessageStart)
    await events.aclose()
    assert recorder.hooks[0] == "start" and recorder.hooks[-1] == "error"
    assert recorder.hooks.count("error") == 1 and "end" not in recorder.hooks


async def test_broken_callback_does_not_change_outcome(backend):
    backend["handler"] = ndjson(TEXT_LINE, DONE_LINE)
    recorder = Recorder(fail=True)
    events = [e async for e in make_provider(backend, recorder).stream(request(stream=True))]
    assert not any(isinstance(e, StreamError) for e in events)
    assert recorder.hooks == ["start", "chunk", "chunk", "end"]


@pytest.mark.parametrize(
    "handler, status",
    [
        (ndjson(json.dumps({"error": "model crashed"})), 502),
        (lambda req: httpx.Response(404, json={"error": "model 'm' not found"}), 404),
    ],
    ids=["error_as_first_line", "refused"],
)
async def test_gateway_stream_failing_before_first_event_is_http_error(backend, handler, status):
    backend["handler"] = handler
    body = {"model": "ollama/m", "messages": [{"role": "user", "content": "x"}], "stream": True}
    resp = await chat_completions(make_provider(backend), body)
    assert resp.status_code == status
    assert "error" in json.loads(resp.body)


async def test_http_clients_pool_one_client_per_prefix():
    seen = []
    transport = httpx.MockTransport(lambda req: seen.append(req.url) or httpx.Response(200))
    clients = HttpClients()
    configs = {
        "vllm": {"base_url": "http://vllm.test/v1", "transport": transport},
        "openai": {"base_url": "http://openai.test/v1", "transport": transport},
    }
    first = provider_for("vllm/a", configs, clients)
    second = provider_for("vllm/b", configs, clients)
    other = provider_for("openai/c", configs, clients)
    assert first.http_client is second.http_client
    assert other.http_client is not first.http_client
    await first.http_client.get("http://vllm.test/v1/models")
    assert seen == [httpx.URL("http://vllm.test/v1/models")]
    await clients.aclose()
    assert first.http_client.is_closed and other.http_client.is_closed
    assert provider_for("vllm/a", configs, clients).http_client is not first.http_client


@pytest.mark.parametrize(
    "model, cls_name, endpoint",
    [
        ("vllm/gpt-oss-20b", "VLLMProvider", "http://vllm.test/v1/chat/completions"),
        ("openai/gpt-5", "OpenAIProvider", "https://api.openai.test/v1/chat/completions"),
        (
            "openrouter/mistralai/mistral-nemo",
            "OpenRouterProvider",
            "https://openrouter.test/api/v1/chat/completions",
        ),
        ("qwen3:4b", "OllamaProvider", "http://ollama.test/api/chat"),
    ],
)
def test_provider_for_routes_prefixes(model, cls_name, endpoint):
    configs = {
        "vllm": {"base_url": "http://vllm.test/v1/"},
        "openai": {"base_url": "https://api.openai.test/v1", "api_key": ""},
        "openrouter": {"base_url": "https://openrouter.test/api/v1", "api_key": "k"},
        "ollama": {"base_url": "http://ollama.test"},
    }
    provider = provider_for(model, configs)
    assert type(provider).__name__ == cls_name
    assert provider.endpoint() == endpoint
    assert provider.model == model.split("/", 1)[-1]


def test_provider_for_empty_api_key_sends_no_auth():
    configs = {"openai": {"base_url": "https://api.openai.test/v1", "api_key": ""}}
    assert provider_for("openai/gpt-5", configs).headers() == {}


def test_provider_for_unported_prefix_is_none():
    assert provider_for("anthropic/claude", {"anthropic": {"base_url": "x"}}) is None


def test_config_headers_go_on_every_request():
    configs = {
        "openai": {
            "base_url": "https://api.openai.test/v1",
            "api_key": "k",
            "headers": {"OpenAI-Organization": "org-1"},
        }
    }
    assert provider_for("openai/gpt-5", configs).headers() == {
        "authorization": "Bearer k",
        "OpenAI-Organization": "org-1",
    }
