"""Gateway regression tests for bugs in backend integration."""

import json

import pytest

from model_calling.callbacks import Callback
from model_calling.core import MessageStart, MessageStop
from tests.harness.replay import fixture_path
from tests.harness.scenarios import BACKENDS, SCENARIOS


class RecordingCallback(Callback):
    """Test callback that records all method calls."""

    def __init__(self):
        self.calls = []

    async def on_request_start(self, model, messages, **kwargs):
        self.calls.append(("on_request_start", model, messages, kwargs))

    async def on_request_end(self, model, messages, response, **kwargs):
        self.calls.append(("on_request_end", model, messages, response, kwargs))

    async def on_stream_chunk(self, model, messages, chunk, **kwargs):
        self.calls.append(("on_stream_chunk", model, messages, chunk, kwargs))

    async def on_error(self, model, messages, error, **kwargs):
        self.calls.append(("on_error", model, messages, error, kwargs))


@pytest.mark.asyncio
async def test_backend_error_returns_exact_status_and_error_body(gateway):
    """Backend 404 used to return HTTP 200. Verify it returns 404 with error body."""
    gateway.use("ollama", "backend_error")

    body = SCENARIOS["backend_error"].chat_completions_body(
        BACKENDS["ollama"].model_for(SCENARIOS["backend_error"])
    )
    resp = await gateway.client.post("/v1/chat/completions", json=body)

    # Status must be the backend's error status, not 200
    assert resp.status_code == 404
    # Body must be an OpenAI error object
    data = resp.json()
    assert "error" in data
    assert "message" in data["error"]
    assert "type" in data["error"]
    assert "not found" in data["error"]["message"].lower()


@pytest.mark.asyncio
async def test_callbacks_run_on_ollama_text(gateway):
    """Callbacks used to be skipped on the Ollama path. Verify they run."""
    callback = RecordingCallback()
    gateway.registry.default_configs["ollama"]["callbacks"] = [callback]
    gateway.use("ollama", "text")

    body = SCENARIOS["text"].chat_completions_body(BACKENDS["ollama"].model_for(SCENARIOS["text"]))
    resp = await gateway.client.post("/v1/chat/completions", json=body)

    assert resp.status_code == 200
    # Should have on_request_start and on_request_end
    assert len(callback.calls) == 2
    assert callback.calls[0][0] == "on_request_start"
    assert callback.calls[1][0] == "on_request_end"

    # Verify model and messages were passed positionally (not in kwargs)
    _, model, messages, kwargs = callback.calls[0]
    assert model == "qwen3:4b"
    assert isinstance(messages, list)
    assert "model" not in kwargs
    assert "messages" not in kwargs


@pytest.mark.asyncio
async def test_callbacks_run_on_backend_error(gateway):
    """on_error callback should be called when the backend fails."""
    callback = RecordingCallback()
    gateway.registry.default_configs["ollama"]["callbacks"] = [callback]
    gateway.use("ollama", "backend_error")

    body = SCENARIOS["backend_error"].chat_completions_body(
        BACKENDS["ollama"].model_for(SCENARIOS["backend_error"])
    )
    resp = await gateway.client.post("/v1/chat/completions", json=body)

    assert resp.status_code == 404
    # Should have on_request_start and on_error
    assert len(callback.calls) == 2
    assert callback.calls[0][0] == "on_request_start"
    assert callback.calls[1][0] == "on_error"

    _, model, messages, error, kwargs = callback.calls[1]
    assert model == BACKENDS["ollama"].missing_native_model
    assert isinstance(error, Exception)


@pytest.mark.asyncio
async def test_callbacks_run_on_stream(gateway):
    """on_stream_chunk gets each internal event; on_request_end the last."""
    callback = RecordingCallback()
    gateway.registry.default_configs["ollama"]["callbacks"] = [callback]
    gateway.use("ollama", "stream")

    body = SCENARIOS["stream"].chat_completions_body(
        BACKENDS["ollama"].model_for(SCENARIOS["stream"])
    )
    resp = await gateway.client.post("/v1/chat/completions", json=body)

    assert resp.status_code == 200
    chunks = [c[3] for c in callback.calls if c[0] == "on_stream_chunk"]
    assert [type(c) for c in chunks[:1]] == [MessageStart]
    assert [type(c) for c in chunks[-1:]] == [MessageStop]
    assert callback.calls[-1][0] == "on_request_end"
    assert callback.calls[-1][3] is chunks[-1]


@pytest.mark.asyncio
async def test_invalid_request_returns_400_with_error_body(gateway):
    """Invalid requests (e.g. n=2) should return 400 with OpenAI error body."""
    replay = gateway.use("ollama", "text")

    body = SCENARIOS["text"].chat_completions_body(BACKENDS["ollama"].model_for(SCENARIOS["text"]))
    body["n"] = 2  # Not supported

    resp = await gateway.client.post("/v1/chat/completions", json=body)

    assert resp.status_code == 400
    data = resp.json()
    assert "error" in data
    assert "message" in data["error"]
    # Rejected before reaching the backend, so the fixture goes unused.
    assert replay.requests == []
    replay._pending.clear()
    assert "type" in data["error"]
    assert data["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize("model", ["anthropic/claude-sonnet-5-5", "cohere/command-a"])
async def test_streaming_unported_backend_is_501_naming_phase_6(gateway, model):
    """Anthropic and Cohere streaming went through a legacy bypass that sent no
    auth and hit the wrong URL. Until Phase 6 ports them it is refused outright."""
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": True}
    resp = await gateway.client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 501, resp.text
    assert "Phase 6" in resp.json()["error"]["message"]


ENVELOPE_ERRORS = {
    "chat_reserved": ("/v1/chat/completions", {"api_base": "http://evil.test"}, "api_base"),
    "messages_reserved": ("/v1/messages", {"api_key": "k", "drop_params": False}, "api_key"),
    "embeddings_reserved": ("/v1/embeddings", {"api_base": "http://evil.test"}, "api_base"),
    "messages_not_object": ("/v1/messages", [], "JSON object"),
    "messages_no_model": ("/v1/messages", {"model": None}, "model is required"),
    "messages_unported": ("/v1/messages", {"model": "cohere/command-a"}, "no provider"),
}


@pytest.mark.parametrize("path, fields, match", ENVELOPE_ERRORS.values(), ids=ENVELOPE_ERRORS)
async def test_envelope_errors_are_400_in_the_ingress_format(gateway, path, fields, match):
    """Reserved Client arguments would point the gateway's key at another host."""
    body = {"model": "ollama/m", "messages": [], "input": "x", "max_tokens": 8}
    body = {**body, **fields} if isinstance(fields, dict) else fields
    resp = await gateway.client.post(path, json=body)
    assert resp.status_code == 400, resp.text
    error = resp.json()["error"]
    assert match in error["message"]
    assert error["type"] == "invalid_request_error"


async def test_messages_stream_error_after_start_is_an_error_event(gateway):
    """Ollama failing mid-stream ends the Anthropic stream with an `error` event."""
    from tests.harness.replay import Exchange, ReplayTransport

    recorded = Exchange.load(fixture_path("ollama", "stream"))
    recorded.lines = recorded.lines[:2] + [json.dumps({"error": "model crashed"})]
    config = gateway.registry.default_configs["ollama"]
    config.update(BACKENDS["ollama"].gateway_config, transport=ReplayTransport([recorded]))
    body = SCENARIOS["stream"].messages_body(BACKENDS["ollama"].gateway_model)
    resp = await gateway.client.post("/v1/messages", json=body)
    assert resp.status_code == 200
    last = resp.text.strip().split("\n\n")[-1]
    assert last.startswith("event: error\n"), last
    assert json.loads(last.split("data: ", 1)[1])["error"] == {
        "type": "api_error",
        "message": "Ollama error: model crashed",
    }


async def test_refresh_reports_capabilities_for_a_vllm_model(gateway):
    """vLLM lost its legacy adapter in Phase 2, and refresh 404ed since."""
    resp = await gateway.client.post("/v1/models/refresh", json={"model": "vllm/gpt-oss-20b"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "refreshed"


FORCED_TOOL = {
    "chat": ("/v1/chat/completions", {"tool_choice": "required"}, "tools"),
    "messages": ("/v1/messages", {"tool_choice": {"type": "any"}}, "anthropic_tools"),
}


def forced_tool_body(choice: dict, tools: str) -> dict:
    scenario = SCENARIOS["tools"]
    return {
        "model": BACKENDS["ollama"].gateway_model,
        "messages": scenario.messages(),
        "max_tokens": 16,
        "tools": list(getattr(scenario, tools)),
        **choice,
    }


@pytest.mark.parametrize("path, choice, tools", FORCED_TOOL.values(), ids=FORCED_TOOL)
async def test_strict_mode_refuses_a_parameter_the_backend_cannot_honor(
    gateway, monkeypatch, path, choice, tools
):
    from model_calling import server

    monkeypatch.setattr(server, "DROP_PARAMS", False)
    resp = await gateway.client.post(path, json=forced_tool_body(choice, tools))
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["message"] == "ollama cannot honor tool_choice"


@pytest.mark.parametrize("path, choice, tools", FORCED_TOOL.values(), ids=FORCED_TOOL)
async def test_default_mode_drops_it_and_sends_the_rest(gateway, path, choice, tools):
    """Without tool_choice the body is exactly the recorded tools request."""
    gateway.use("ollama", "tools")
    body = {**forced_tool_body(choice, tools), "max_tokens": SCENARIOS["tools"].max_tokens}
    resp = await gateway.client.post(path, json=body)
    assert resp.status_code == 200, resp.text


@pytest.mark.parametrize(
    "value, strict",
    [(None, False), ("", False), ("false", False), ("0", False), ("TRUE", True), ("1", True)],
)
def test_strict_params_setting(value, strict):
    from model_calling.server import STRICT_PARAMS_ENV, strict_params

    environ = {} if value is None else {STRICT_PARAMS_ENV: value}
    assert strict_params(environ) is strict


def test_strict_params_setting_rejects_other_values():
    from model_calling.server import STRICT_PARAMS_ENV, strict_params

    with pytest.raises(RuntimeError, match=STRICT_PARAMS_ENV):
        strict_params({STRICT_PARAMS_ENV: "maybe"})
