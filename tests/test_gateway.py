"""Gateway regression tests for bugs in backend integration."""

import json

import pytest

from model_calling.callbacks import Callback
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
    """on_stream_chunk should be called for each NDJSON line."""
    import json

    callback = RecordingCallback()
    gateway.registry.default_configs["ollama"]["callbacks"] = [callback]
    gateway.use("ollama", "stream")

    body = SCENARIOS["stream"].chat_completions_body(
        BACKENDS["ollama"].model_for(SCENARIOS["stream"])
    )
    resp = await gateway.client.post("/v1/chat/completions", json=body)

    assert resp.status_code == 200
    # Count on_stream_chunk calls
    chunk_calls = [c for c in callback.calls if c[0] == "on_stream_chunk"]

    # Load the fixture to count expected lines
    fixture = json.loads(fixture_path("ollama", "stream").read_text())
    expected_lines = len(fixture["response"]["lines"])

    assert len(chunk_calls) == expected_lines


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
