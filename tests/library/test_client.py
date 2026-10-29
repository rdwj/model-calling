"""
`Client` behavior beyond the grid: routing and config, callbacks, the sync
API, stream lifecycle, and the module-level functions.
"""

import asyncio
import json
import re
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import httpx
import pytest

import model_calling.library as library_module
from model_calling.callbacks import Callback
from model_calling.errors import (
    APIError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)
from model_calling.library import Client
from model_calling.library._sync import ClosedError
from model_calling.types.chat import ChatCompletion
from model_calling.types.messages import Message
from tests.harness.replay import Exchange, ReplayTransport, fixture_path
from tests.harness.scenarios import BACKENDS, SCENARIOS, VLLM_URL

OLLAMA = BACKENDS["ollama"]
REPO_ROOT = Path(__file__).parents[2]
TEXT_LINE = json.dumps({"model": "m", "message": {"role": "assistant", "content": "hi"}})
USER = [{"role": "user", "content": "x"}]


class Recorder(Callback):
    def __init__(self, name="cb", log=None):
        self.name = name
        self.log = [] if log is None else log

    async def on_request_start(self, model, messages, **kwargs):
        self.log.append((self.name, "start"))

    async def on_request_end(self, model, messages, response, **kwargs):
        self.log.append((self.name, "cancelled" if kwargs.get("cancelled") else "end"))

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
    [("cohere/x", "no provider for 'cohere/x'"), ("bogus/x", "no provider")],
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
    assert log == [("cb", "start"), ("cb", "cancelled")]


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


# --- strict by default ------------------------------------------------------

CACHED_TEXT = {"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}
CACHED_TOOL = {
    "name": "get_weather",
    "input_schema": {"type": "object"},
    "cache_control": {"type": "ephemeral"},
}
STRICT_CASES = {
    "ollama_forced_tool": (
        "completion",
        "ollama/m",
        {"tools": [SCENARIOS["tools"].tools[0]], "tool_choice": "required"},
        "tool_choice",
    ),
    "vllm_thinking_budget": (
        "messages",
        "vllm/m",
        {"thinking": {"type": "enabled", "budget_tokens": 1024}},
        "thinking.budget_tokens",
    ),
}
# A caching hint changes no output, so every backend drops it even in strict mode.
CACHE_CASES = {
    "ollama_cached_block": ("ollama/m", {"messages": [{"role": "user", "content": [CACHED_TEXT]}]}),
    "ollama_cached_tool": ("ollama/m", {"tools": [CACHED_TOOL]}),
    "openrouter_cached_system": ("openrouter/m", {"system": [CACHED_TEXT]}),
    "vllm_cached_tool": ("vllm/m", {"tools": [CACHED_TOOL]}),
}


async def call(client, family, model, params, **kwargs):
    params = {"messages": USER, **params}
    if family == "messages":
        return await client.amessages(model, max_tokens=16, **params, **kwargs)
    return await client.acompletion(model, **params, **kwargs)


def strict_client(transport, **kwargs) -> Client:
    config = {
        p: {"transport": transport, "api_key": "k"}
        for p in ("ollama", "vllm", "openai", "openrouter")
    }
    return Client(config=config, **kwargs)


@pytest.mark.parametrize(
    "family, model, params, name", STRICT_CASES.values(), ids=STRICT_CASES.keys()
)
async def test_unhonorable_parameter_is_refused(family, model, params, name):
    sent = []
    async with strict_client(mock(lambda req: sent.append(req) or httpx.Response(500))) as client:
        with pytest.raises(BadRequestError, match=rf"cannot honor {re.escape(name)}.*drop_params"):
            await call(client, family, model, params)
    assert sent == []


@pytest.mark.parametrize("where", ["client", "call"])
@pytest.mark.parametrize(
    "family, model, params, name", STRICT_CASES.values(), ids=STRICT_CASES.keys()
)
async def test_drop_params_sends_without_the_parameter(family, model, params, name, where):
    sent = []
    transport = mock(lambda req: sent.append(json.loads(req.content)) or httpx.Response(500))
    client_kwargs, call_kwargs = (
        ({"drop_params": True}, {}) if where == "client" else ({}, {"drop_params": True})
    )
    async with strict_client(transport, **client_kwargs) as client:
        with pytest.raises(InternalServerError):
            await call(client, family, model, params, **call_kwargs)
    top_level = name.split(".")[0]
    assert top_level not in json.dumps(sent[0]), sent[0]


@pytest.mark.parametrize("model, params", CACHE_CASES.values(), ids=CACHE_CASES.keys())
async def test_cache_control_is_dropped_silently_in_strict_mode(model, params):
    sent = []
    transport = mock(lambda req: sent.append(json.loads(req.content)) or httpx.Response(500))
    async with strict_client(transport) as client:
        with pytest.raises(InternalServerError):
            await call(client, "messages", model, params)
    assert len(sent) == 1 and "cache_control" not in json.dumps(sent[0]), sent


@pytest.mark.parametrize("stream", [False, True])
async def test_request_the_backend_cannot_express_is_bad_request(stream):
    """OpenAI's provider rejects top_k while building the native body."""
    async with strict_client(mock(lambda req: httpx.Response(500))) as client:
        with pytest.raises(BadRequestError, match="top_k") as err:
            await client.acompletion("openai/m", USER, top_k=5, stream=stream)
    assert err.value.provider == "openai"


# --- review regressions: sync shutdown, typed stream errors -----------------


def slow_stream(started: threading.Event):
    """An Ollama stream that sends one line, then stalls."""

    class Stalled(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield (TEXT_LINE + "\n").encode()
            started.set()
            await asyncio.sleep(60)

    return mock(lambda req: httpx.Response(200, stream=Stalled()))


def test_close_with_an_open_sync_stream_finishes_it():
    log = []
    client = ollama_client(slow_stream(threading.Event()), callbacks=[Recorder(log=log)])
    chunks = client.completion("ollama/m", USER, stream=True)
    next(chunks)
    client.close()
    assert log == [("cb", "start"), ("cb", "cancelled")]
    with pytest.raises(ClosedError):
        next(chunks)


def test_close_cancels_a_call_in_flight_on_another_thread():
    started, outcome = threading.Event(), []

    def worker():
        try:
            for _ in client.completion("ollama/m", USER, stream=True):
                pass
        except BaseException as e:  # noqa: BLE001 - recording whatever ends the call
            outcome.append(e)

    client = ollama_client(slow_stream(started))
    thread = threading.Thread(target=worker)
    thread.start()
    assert started.wait(5)
    client.close()
    thread.join(5)
    assert not thread.is_alive(), "worker hung after close()"
    assert outcome, "worker finished without an error"


def test_exit_with_an_open_sync_stream_does_not_hang():
    script = textwrap.dedent("""
        import json, httpx
        from model_calling import Client
        line = json.dumps({"model": "m", "message": {"role": "assistant", "content": "hi"}})
        lines = (line + "\\n") * 3
        transport = httpx.MockTransport(lambda req: httpx.Response(200, text=lines))
        client = Client(config={"ollama": {"transport": transport}})
        chunks = client.completion("ollama/m", [{"role": "user", "content": "x"}], stream=True)
        next(chunks)
        print("exiting")
        """)
    done = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
    )
    assert (done.returncode, done.stdout.strip()) == (0, "exiting"), done.stderr


@pytest.mark.parametrize(
    "kind, error_type", [("rate_limit_error", RateLimitError), (None, APIError)]
)
async def test_mid_stream_error_kind_picks_the_class(kind, error_type):
    error = {"message": "slow down", **({"type": kind} if kind else {})}
    sse = f"data: {json.dumps({'error': error})}\n\n"
    transport = mock(
        lambda req: httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
    )
    async with strict_client(transport) as client:
        with pytest.raises(APIError, match="slow down") as err:
            await client.acompletion("openrouter/m", USER, stream=True)
    assert type(err.value) is error_type


# --- typed content blocks as input -----------------------------------------

TOOL_RESULT_TEXT = "22C and sunny"


def conversation(first: Message | None) -> list:
    """The tools scenario, then (given its answer) the assistant turn as the
    typed blocks `messages()` returned, and a tool result."""
    turns = list(SCENARIOS["tools"].messages_body("ollama/m")["messages"])
    if first is None:
        return turns
    call = next(b for b in first.content if b.type == "tool_use")
    result = {"type": "tool_result", "tool_use_id": call.id, "content": TOOL_RESULT_TEXT}
    return turns + [
        {"role": "assistant", "content": first.content},
        {"role": "user", "content": [result]},
    ]


def tools_backend():
    """Answers every request with the recorded Ollama tools response
    (thinking, then a call), keeping the bodies sent."""
    recorded = Exchange.load(fixture_path("ollama", "tools")).response_json
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=recorded)

    return mock(handler), sent, recorded["message"]


def assert_follow_up(sent: list, recorded: dict):
    """The second request carries the first answer's thinking and call back."""
    _, assistant, tool = sent[1]["messages"]
    assert assistant == {
        "role": "assistant",
        "content": "",
        "thinking": recorded["thinking"],
        "tool_calls": [{"function": {"name": "get_weather", "arguments": {"location": "Paris"}}}],
    }
    assert tool == {"role": "tool", "content": TOOL_RESULT_TEXT, "tool_name": "get_weather"}


TOOLS = list(SCENARIOS["tools"].anthropic_tools)


async def test_amessages_takes_its_own_content_back():
    transport, sent, recorded = tools_backend()
    async with ollama_client(transport) as client:
        first = await client.amessages("ollama/m", conversation(None), max_tokens=256, tools=TOOLS)
        assert [b.type for b in first.content] == ["thinking", "tool_use"]
        await client.amessages("ollama/m", conversation(first), max_tokens=256, tools=TOOLS)
    assert_follow_up(sent, recorded)


def test_messages_takes_its_own_content_back():
    transport, sent, recorded = tools_backend()
    with ollama_client(transport) as client:
        first = client.messages("ollama/m", conversation(None), max_tokens=256, tools=TOOLS)
        turns = conversation(first)
        client.messages("ollama/m", turns, max_tokens=256, tools=TOOLS)
    assert_follow_up(sent, recorded)
    assert turns[1]["content"] is first.content, "the caller's turns were replaced"


async def test_blocks_the_caller_builds_keep_their_type():
    from model_calling.types.messages import TextBlock

    transport, sent, _ = tools_backend()
    async with ollama_client(transport) as client:
        turn = {"role": "user", "content": [TextBlock(text="hi"), TextBlock(text="there")]}
        await client.amessages("ollama/m", [turn], max_tokens=256)
    assert sent[0]["messages"] == [{"role": "user", "content": "hi\nthere"}]


async def test_dict_blocks_and_typed_blocks_send_the_same_body():
    transport, sent, _ = tools_backend()
    async with ollama_client(transport) as client:
        first = await client.amessages("ollama/m", conversation(None), max_tokens=256, tools=TOOLS)
        typed = conversation(first)
        as_dicts = [*typed[:1], {**typed[1], "content": [b.to_dict() for b in first.content]}]
        as_dicts.append(typed[2])
        await client.amessages("ollama/m", typed, max_tokens=256, tools=TOOLS)
        await client.amessages("ollama/m", as_dicts, max_tokens=256, tools=TOOLS)
    assert sent[1] == sent[2]
