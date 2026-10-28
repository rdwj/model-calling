"""
Shared provider HTTP behavior: error mapping, stream lifecycle, callbacks.

Exercised through OllamaProvider against an httpx.MockTransport.
"""

import json

import httpx
import pytest

from model_calling.callbacks import Callback
from model_calling.core import Message, MessageStart, Request, StreamError, Text
from model_calling.gateway import chat_completions
from model_calling.providers import BackendError, OllamaProvider

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
def backend(monkeypatch):
    """Route every httpx.AsyncClient to a handler the test sets."""
    state = {}
    real_init = httpx.AsyncClient.__init__

    def init(client, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(lambda req: state["handler"](req))
        real_init(client, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)
    return state


def make_provider(*callbacks):
    return OllamaProvider(model="m", base_url="http://ollama.test", callbacks=callbacks)


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


@pytest.mark.parametrize(
    "handler, status",
    [
        (lambda req: httpx.Response(200, text="<html>not json</html>"), 502),
        (lambda req: httpx.Response(500, json={"error": "boom"}), 502),
        (lambda req: httpx.Response(429, json={"error": "slow down"}), 429),
        (
            lambda req: httpx.Response(
                200,
                json={
                    "model": "m",
                    "message": {"tool_calls": [{"function": {"arguments": {}}}]},
                    "done": True,
                },
            ),
            502,
        ),
    ],
    ids=["non_json", "server_error", "rate_limited", "unreadable_tool_call"],
)
async def test_complete_maps_backend_failures(backend, handler, status):
    backend["handler"] = handler
    recorder = Recorder()
    with pytest.raises(BackendError) as err:
        await make_provider(recorder).complete(request())
    assert err.value.status == status
    assert recorder.hooks == ["start", "error"]


async def test_stream_mid_stream_transport_error_is_backend_error(backend):
    backend["handler"] = broken_stream
    recorder = Recorder()
    events = make_provider(recorder).stream(request(stream=True))
    with pytest.raises(BackendError, match="stream broke"):
        async for _ in events:
            pass
    assert recorder.hooks[-1] == "error"


async def test_stream_error_line_is_event_and_error_callback(backend):
    backend["handler"] = ndjson(TEXT_LINE, json.dumps({"error": "model crashed"}))
    recorder = Recorder()
    events = [e async for e in make_provider(recorder).stream(request(stream=True))]
    assert isinstance(events[-1], StreamError)
    assert recorder.hooks == ["start", "chunk", "chunk", "error"]


async def test_consumer_closing_early_still_ends_callbacks(backend):
    backend["handler"] = ndjson(TEXT_LINE, TEXT_LINE, DONE_LINE)
    recorder = Recorder()
    events = make_provider(recorder).stream(request(stream=True))
    assert isinstance(await anext(events), MessageStart)
    await events.aclose()
    assert recorder.hooks[0] == "start" and recorder.hooks[-1] == "error"
    assert recorder.hooks.count("error") == 1 and "end" not in recorder.hooks


async def test_broken_callback_does_not_change_outcome(backend):
    backend["handler"] = ndjson(TEXT_LINE, DONE_LINE)
    recorder = Recorder(fail=True)
    events = [e async for e in make_provider(recorder).stream(request(stream=True))]
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
    resp = await chat_completions(make_provider(), body)
    assert resp.status_code == status
    assert "error" in json.loads(resp.body)
