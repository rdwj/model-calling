"""
`Client` behavior beyond the grid: routing and config, callbacks, the sync
API, stream lifecycle, and the module-level functions.
"""

import asyncio
import json

import httpx
import pytest

import model_calling.library as library_module
from model_calling.callbacks import Callback
from model_calling.errors import APIError, BadRequestError, NotFoundError
from model_calling.library import Client
from model_calling.types.chat import ChatCompletion
from model_calling.types.messages import Message
from tests.harness.replay import Exchange, ReplayTransport, fixture_path
from tests.harness.scenarios import BACKENDS, SCENARIOS, VLLM_URL

OLLAMA = BACKENDS["ollama"]
TEXT_LINE = json.dumps({"model": "m", "message": {"role": "assistant", "content": "hi"}})
USER = [{"role": "user", "content": "x"}]


class Recorder(Callback):
    def __init__(self, name="cb", log=None):
        self.name = name
        self.log = [] if log is None else log

    async def on_request_start(self, model, messages, **kwargs):
        self.log.append((self.name, "start"))

    async def on_request_end(self, model, messages, response, **kwargs):
        self.log.append((self.name, "end"))

    async def on_error(self, model, messages, error, **kwargs):
        self.log.append((self.name, "error"))


def replay(*scenarios, backend="ollama") -> ReplayTransport:
    return ReplayTransport([Exchange.load(fixture_path(backend, s)) for s in scenarios])


def ollama_client(transport, **kwargs) -> Client:
    return Client(config={"ollama": {**OLLAMA.gateway_config, "transport": transport}}, **kwargs)


def body(scenario: str, backend=OLLAMA) -> dict:
    return SCENARIOS[scenario].chat_completions_body(backend.model_for(SCENARIOS[scenario]))


def mock(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


@pytest.mark.parametrize(
    "model, match",
    [("anthropic/claude-x", "no provider for 'anthropic/claude-x'"), ("cohere/x", "no provider")],
)
async def test_unported_prefix_is_bad_request(model, match):
    async with Client() as client:
        with pytest.raises(BadRequestError, match=match) as err:
            await client.acompletion(model, USER)
    assert err.value.status is None


async def test_invalid_request_is_rejected_before_sending():
    transport = replay("text")
    async with ollama_client(transport) as client:
        with pytest.raises(BadRequestError) as err:
            await client.acompletion(**body("text"), n=2)
    assert err.value.provider == "ollama"
    assert transport.requests == []


async def test_missing_base_url_is_bad_request():
    async with Client(config={"ollama": {"base_url": ""}}) as client:
        with pytest.raises(BadRequestError, match="no base_url"):
            await client.acompletion("ollama/m", USER)


async def test_per_call_api_base_and_key_override_config():
    transport = replay("text", backend="vllm")
    config = {
        "vllm": {
            "base_url": "http://elsewhere.test/v1",
            "api_key": "config-key",
            "transport": transport,
        }
    }
    async with Client(config=config) as client:
        await client.acompletion(
            **body("text", BACKENDS["vllm"]), api_base=VLLM_URL, api_key="call-key"
        )
    assert transport.requests[0].headers["authorization"] == "Bearer call-key"


async def test_client_callbacks_run_before_prefix_callbacks():
    log = []
    transport = replay("text")
    config = {
        "ollama": {
            **OLLAMA.gateway_config,
            "transport": transport,
            "callbacks": [Recorder("prefix", log)],
        }
    }
    async with Client(config=config, callbacks=[Recorder("client", log)]) as client:
        await client.acompletion(**body("text"))
    assert log == [("client", "start"), ("prefix", "start"), ("client", "end"), ("prefix", "end")]


async def test_stream_refusal_raises_on_await_not_mid_loop():
    transport = mock(lambda req: httpx.Response(404, json={"error": "model 'm' not found"}))
    async with ollama_client(transport) as client:
        with pytest.raises(NotFoundError):
            await client.acompletion("ollama/m", USER, stream=True)


async def test_mid_stream_backend_error_raises():
    lines = [TEXT_LINE, json.dumps({"error": "model crashed"})]
    transport = mock(lambda req: httpx.Response(200, text="\n".join(lines) + "\n"))
    async with ollama_client(transport) as client:
        chunks = await client.acompletion("ollama/m", USER, stream=True)
        with pytest.raises(APIError, match="model crashed") as err:
            async for _ in chunks:
                pass
    assert (err.value.provider, err.value.status) == ("ollama", None)


async def test_stream_closed_early_ends_with_one_callback():
    log = []
    async with ollama_client(replay("stream"), callbacks=[Recorder(log=log)]) as client:
        chunks = await client.acompletion(**body("stream"))
        async for _ in chunks:
            break
        await chunks.aclose()
    assert log[0] == ("cb", "start")
    assert [entry for entry in log[1:]] in ([("cb", "end")], [("cb", "error")])


def test_sync_completion_and_stream():
    with ollama_client(replay("text", "stream")) as client:
        resp = client.completion(**body("text"))
        assert isinstance(resp, ChatCompletion)
        assert "pong" in resp.choices[0].message.content.lower()
        chunks = client.completion(**body("stream"))
        assert not hasattr(chunks, "__aiter__")
        text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert "pong" in text.lower()


def test_sync_messages():
    scenario = SCENARIOS["text"]
    with ollama_client(replay("text")) as client:
        msg = client.messages(**scenario.messages_body(OLLAMA.model_for(scenario)))
    assert isinstance(msg, Message)
    assert msg.stop_reason == "end_turn"


def test_sync_stream_refusal_raises_at_call():
    transport = mock(lambda req: httpx.Response(404, json={"error": "model 'm' not found"}))
    with ollama_client(transport) as client:
        with pytest.raises(NotFoundError):
            client.completion("ollama/m", USER, stream=True)


def test_one_client_serves_successive_event_loops():
    """A script calling asyncio.run twice must not reuse a dead loop's connections."""
    client = ollama_client(replay("text", "text"))
    for _ in range(2):
        resp = asyncio.run(client.acompletion(**body("text")))
        assert resp.choices[0].finish_reason == "stop"
    client.close()


def test_sync_call_from_a_callback_is_refused():
    class Reentrant(Callback):
        def __init__(self, client_ref):
            self.client_ref = client_ref
            self.error = None

        async def on_request_start(self, model, messages, **kwargs):
            try:
                self.client_ref[0].completion(**body("text"))
            except RuntimeError as e:
                self.error = e

    ref = []
    callback = Reentrant(ref)
    with ollama_client(replay("text"), callbacks=[callback]) as client:
        ref.append(client)
        client.completion(**body("text"))
    assert "use the async API" in str(callback.error)


async def test_module_level_functions_use_the_default_client(monkeypatch):
    client = ollama_client(replay("text"))
    monkeypatch.setattr(library_module, "_default", client)
    resp = await library_module.acompletion(**body("text"))
    assert resp.choices[0].finish_reason == "stop"
    await client.aclose()


def test_default_client_is_created_once(monkeypatch):
    monkeypatch.setattr(library_module, "_default", None)
    first = library_module.default_client()
    assert library_module.default_client() is first
    first.close()
