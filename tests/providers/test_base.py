"""
Shared provider HTTP behavior: error mapping, stream lifecycle, callbacks.

Exercised through OllamaProvider against an httpx.MockTransport.
"""

import asyncio
import json

import httpx
import pytest

from model_calling.callbacks import Callback
from model_calling.core import Message, MessageStart, Request, StreamError, Text
from model_calling.core.embeddings import EmbeddingRequest
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
from model_calling.library import Client
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
        await self._record("cancelled" if kwargs.get("cancelled") else "end")

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


def broken_stream(req, error=httpx.ReadError("connection reset")):
    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield (TEXT_LINE + "\n").encode()
            raise error

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
        # No backend status: a library refusal, or an error reported in a stream.
        (BadRequestError("x"), 400),
        (NotFoundError("x"), 404),
        (RateLimitError("x"), 429),
        (AuthenticationError("x"), 502),
        (InternalServerError("x"), 502),
    ],
    ids=lambda v: type(v).__name__ if isinstance(v, Exception) else str(v),
)
def test_gateway_client_status(error, status):
    """The gateway hides backend 5xx and credential refusals behind a 502."""
    assert client_status(error) == status


@pytest.mark.parametrize(
    "error",
    [httpx.ReadError("connection reset"), httpx.ReadTimeout("stalled")],
    ids=["reset", "stall"],
)
async def test_stream_mid_stream_transport_error_is_connection_error(backend, error):
    """A stall after the stream starts is a broken connection, which the gateway answers 502."""
    backend["handler"] = lambda req: broken_stream(req, error)
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
    # One chunk per event the consumer saw before the error.
    assert recorder.hooks == ["start", *["chunk"] * (len(events) - 1), "error"]


async def test_consumer_closing_early_ends_as_cancelled(backend):
    """Stopping early is the consumer's choice, not a failure (decided 2026-10-28)."""
    backend["handler"] = ndjson(TEXT_LINE, TEXT_LINE, DONE_LINE)
    recorder = Recorder()
    events = make_provider(backend, recorder).stream(request(stream=True))
    assert isinstance(await anext(events), MessageStart)
    await events.aclose()
    assert recorder.hooks == ["start", "chunk", "cancelled"]


def stalled_stream(req, started: asyncio.Event | None = None) -> httpx.Response:
    """A stream that sends one line, then stalls."""

    class Stalled(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield (TEXT_LINE + "\n").encode()
            if started is not None:
                started.set()
            await asyncio.sleep(60)

    return httpx.Response(200, stream=Stalled())


async def test_cancelled_consumer_task_ends_as_cancelled(backend):
    started = asyncio.Event()
    backend["handler"] = lambda req: stalled_stream(req, started)
    recorder = Recorder()

    async def consume():
        async for _ in make_provider(backend, recorder).stream(request(stream=True)):
            pass

    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert recorder.hooks[0] == "start" and recorder.hooks[-1] == "cancelled"
    assert "error" not in recorder.hooks


async def test_cancel_scope_still_lets_the_cancelled_callback_finish(backend):
    """Starlette cancels a disconnected client's stream through an anyio cancel
    scope, which cancels every later await too, the callback's included."""
    import anyio

    log = []

    class Slow(Callback):
        async def on_request_end(self, model, messages, response, **kwargs):
            log.append("end-start")
            await asyncio.sleep(0.01)
            log.append(("end-done", kwargs.get("cancelled")))

    backend["handler"] = lambda req: stalled_stream(req)
    events = make_provider(backend, Slow()).stream(request(stream=True))

    async def consume():
        async for _ in events:
            pass

    async with anyio.create_task_group() as group:
        group.start_soon(consume)
        await asyncio.sleep(0.05)
        group.cancel_scope.cancel()
    await asyncio.sleep(0.05)
    assert log == ["end-start", ("end-done", True)]


async def test_chunk_callbacks_see_internal_events(backend):
    seen = []

    class Chunks(Callback):
        async def on_stream_chunk(self, model, messages, chunk, **kwargs):
            seen.append(chunk)

        async def on_request_end(self, model, messages, response, **kwargs):
            seen.append(("end", response))

    backend["handler"] = ndjson(TEXT_LINE, DONE_LINE)
    events = [e async for e in make_provider(backend, Chunks()).stream(request(stream=True))]
    assert seen == [*events, ("end", events[-1])]


async def test_broken_callback_does_not_change_outcome(backend):
    backend["handler"] = ndjson(TEXT_LINE, DONE_LINE)
    recorder = Recorder(fail=True)
    events = [e async for e in make_provider(backend, recorder).stream(request(stream=True))]
    assert not any(isinstance(e, StreamError) for e in events)
    assert recorder.hooks == ["start", *["chunk"] * len(events), "end"]


@pytest.mark.parametrize(
    "handler, status",
    [
        (ndjson(json.dumps({"error": "model crashed"})), 502),
        (lambda req: httpx.Response(404, json={"error": "model 'm' not found"}), 404),
    ],
    ids=["error_as_first_line", "refused"],
)
async def test_gateway_stream_failing_before_first_event_is_http_error(handler, status):
    from model_calling.aliases import parse

    aliases = parse({"models": {"m": {"backend": "ollama", "model": "m"}}})
    body = {"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": True}
    config = {"ollama": {"transport": httpx.MockTransport(handler)}}
    async with Client(config=config) as client:
        resp = await chat_completions(client, body, aliases, drop_params=True)
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


def test_provider_for_unknown_prefix_is_none():
    assert provider_for("bogus/command-a", {"bogus": {"base_url": "x"}}) is None


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


def test_http_clients_replace_a_client_from_another_loop():
    import asyncio

    clients = HttpClients()

    async def get():
        return clients.get("ollama", {})

    first, second = asyncio.run(get()), asyncio.run(get())
    assert first is not second


async def test_http_clients_reuse_on_the_same_loop():
    clients = HttpClients()
    assert clients.get("ollama", {}) is clients.get("ollama", {})
    await clients.aclose()


@pytest.mark.parametrize(
    "environ, prefix, expected",
    [
        ({}, "ollama", {"base_url": "http://localhost:11434"}),
        ({"OPENROUTER_API_KEY": "k"}, "openrouter", {"api_key": "k"}),
        ({"OPENAI_ORG_ID": "org"}, "openai", {"headers": {"OpenAI-Organization": "org"}}),
        ({}, "openai", {"api_key": "", "headers": {}}),
    ],
)
def test_default_configs_read_only_the_given_environment(environ, prefix, expected):
    from model_calling.providers import default_configs

    config = default_configs(environ)[prefix]
    assert expected.items() <= config.items()


# Every prefix, with a base URL whose 404 message must name the request URL.
NOT_FOUND_KEY = "key-that-must-not-leak"
NOT_FOUND_CONFIGS = {
    "anthropic": {"base_url": "https://anthropic.test/v1", "api_key": NOT_FOUND_KEY},
    "cohere": {"base_url": "https://cohere.test/v2", "api_key": NOT_FOUND_KEY},
    "ollama": {"base_url": "http://ollama.test"},
    "openai": {"base_url": "https://openai.test/v1", "api_key": NOT_FOUND_KEY},
    "openrouter": {"base_url": "https://openrouter.test/api/v1", "api_key": NOT_FOUND_KEY},
    "openrouter_responses": {
        "base_url": "https://openrouter.test/api/v1",
        "api_key": NOT_FOUND_KEY,
    },
    "vllm": {"base_url": "http://vllm.test/v1"},
}
EMBEDDING_PREFIXES = ("cohere", "ollama", "openai", "vllm")


def not_found_provider(prefix: str):
    """A `prefix` provider whose backend answers every request with a bare 404."""
    transport = httpx.MockTransport(lambda req: httpx.Response(404, text="404 page not found"))
    config = {**NOT_FOUND_CONFIGS[prefix], "transport": transport}
    return provider_for(f"{prefix}/m", {prefix: config}, HttpClients())


async def drain(events):
    return [e async for e in events]


@pytest.mark.parametrize(
    "prefix, call",
    [(prefix, call) for prefix in sorted(NOT_FOUND_CONFIGS) for call in ("complete", "stream")]
    + [(prefix, "embed") for prefix in EMBEDDING_PREFIXES],
)
async def test_a_404_names_the_request_url(prefix, call):
    provider = not_found_provider(prefix)
    if call == "embed":
        url = provider.embedding_endpoint()
        pending = provider.embed(EmbeddingRequest(model=f"{prefix}/m", input=["x"]))
    else:
        url = provider.endpoint()
        req = request(stream=call == "stream")
        req = req.model_copy(update={"model": f"{prefix}/m", "max_tokens": 8})
        pending = provider.complete(req) if call == "complete" else drain(provider.stream(req))
    with pytest.raises(NotFoundError) as err:
        await pending
    assert err.value.message == f"{prefix} returned HTTP 404 for POST {url}: 404 page not found"
    # Keys travel in headers, so naming the URL can't leak one.
    assert NOT_FOUND_KEY not in err.value.message


async def test_other_statuses_leave_the_url_out(backend):
    backend["handler"] = lambda req: httpx.Response(500, json={"error": "boom"})
    with pytest.raises(InternalServerError) as err:
        await make_provider(backend).complete(request())
    assert err.value.message == "ollama returned HTTP 500: boom"
