"""Gateway regression tests for bugs in backend integration."""

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

    body = SCENARIOS["backend_error"].chat_completions_body(BACKENDS["ollama"].gateway_model)
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

    body = SCENARIOS["text"].chat_completions_body(BACKENDS["ollama"].gateway_model)
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

    body = SCENARIOS["backend_error"].chat_completions_body(BACKENDS["ollama"].gateway_model)
    resp = await gateway.client.post("/v1/chat/completions", json=body)

    assert resp.status_code == 404
    # Should have on_request_start and on_error
    assert len(callback.calls) == 2
    assert callback.calls[0][0] == "on_request_start"
    assert callback.calls[1][0] == "on_error"

    _, model, messages, error, kwargs = callback.calls[1]
    assert model == "qwen3:4b"
    assert isinstance(error, Exception)


@pytest.mark.asyncio
async def test_callbacks_run_on_stream(gateway):
    """on_stream_chunk should be called for each NDJSON line."""
    import json

    callback = RecordingCallback()
    gateway.registry.default_configs["ollama"]["callbacks"] = [callback]
    gateway.use("ollama", "stream")

    body = SCENARIOS["stream"].chat_completions_body(BACKENDS["ollama"].gateway_model)
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
    gateway.use("ollama", "text")

    body = SCENARIOS["text"].chat_completions_body(BACKENDS["ollama"].gateway_model)
    body["n"] = 2  # Not supported

    resp = await gateway.client.post("/v1/chat/completions", json=body)

    assert resp.status_code == 400
    data = resp.json()
    assert "error" in data
    assert "message" in data["error"]
    assert "type" in data["error"]
    assert data["error"]["type"] == "invalid_request_error"
