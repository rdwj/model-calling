"""Internal stream events to Anthropic Messages chunks."""

import json

import pytest

from model_calling.core import (
    BlockDelta,
    BlockStart,
    BlockStop,
    MessageDelta,
    MessageStart,
    MessageStop,
    StreamError,
    Text,
    TextDelta,
    ToolCall,
    ToolInputDelta,
    Usage,
)
from model_calling.formats.anthropic_stream import AnthropicStreamEncoder
from model_calling.providers import OllamaProvider, OpenRouterProvider, VLLMProvider
from tests.harness.sdk import validate_anthropic_stream


def _parse_events(encoder, provider, fixture_path):
    """Parse a fixture's stream lines through the provider parser and encoder."""
    with open(fixture_path) as f:
        fixture = json.load(f)
    parser = provider.stream_parser()
    events = []
    for line in fixture["response"]["lines"]:
        if line.strip():
            events.extend(parser.feed(line))
    events.extend(parser.close())
    chunks = []
    for event in events:
        chunks.extend(encoder.encode(event))
    return chunks


@pytest.mark.parametrize(
    "fixture,provider_cls,model",
    [
        (
            "tests/fixtures/ollama/stream.json",
            OllamaProvider,
            "qwen3:4b",
        ),
        (
            "tests/fixtures/vllm/stream.json",
            VLLMProvider,
            "gpt-oss-20b",
        ),
        (
            "tests/fixtures/openai_compat/stream.json",
            OpenRouterProvider,
            "mistralai/mistral-nemo",
        ),
    ],
)
def test_encode_backend_stream(fixture, provider_cls, model):
    """Each backend's stream fixture encodes to valid Anthropic events."""
    encoder = AnthropicStreamEncoder(model)
    provider = provider_cls(model=model, base_url="http://test")
    chunks = _parse_events(encoder, provider, fixture)
    validate_anthropic_stream(chunks)
    # Basic grammar checks
    assert chunks[0]["type"] == "message_start"
    assert chunks[-1]["type"] == "message_stop"
    message_delta_count = sum(1 for c in chunks if c["type"] == "message_delta")
    assert message_delta_count == 1, "expected one message_delta before message_stop"
    # Check that message model matches what was passed to the encoder
    assert chunks[0]["message"]["model"] == model


@pytest.mark.parametrize(
    "fixture,provider_cls,model",
    [
        (
            "tests/fixtures/ollama/stream.json",
            OllamaProvider,
            "qwen3:4b",
        ),
        (
            "tests/fixtures/vllm/stream.json",
            VLLMProvider,
            "gpt-oss-20b",
        ),
        (
            "tests/fixtures/openai_compat/stream.json",
            OpenRouterProvider,
            "mistralai/mistral-nemo",
        ),
    ],
)
def test_stream_grammar(fixture, provider_cls, model):
    """Stream grammar holds: start/stop properly nested, indexes correct."""
    encoder = AnthropicStreamEncoder(model)
    provider = provider_cls(model=model, base_url="http://test")
    chunks = _parse_events(encoder, provider, fixture)

    # Track block state
    seen_message_start = False
    seen_message_delta = False
    seen_message_stop = False
    block_starts = {}
    block_stops = set()

    for chunk in chunks:
        t = chunk["type"]
        if t == "message_start":
            assert not seen_message_start, "duplicate message_start"
            seen_message_start = True
        elif t == "content_block_start":
            idx = chunk["index"]
            assert idx not in block_starts, f"duplicate block_start for index {idx}"
            block_starts[idx] = True
        elif t == "content_block_delta":
            idx = chunk["index"]
            assert idx in block_starts, f"delta for block {idx} before start"
            assert idx not in block_stops, f"delta for block {idx} after stop"
        elif t == "content_block_stop":
            idx = chunk["index"]
            assert idx in block_starts, f"stop for block {idx} before start"
            assert idx not in block_stops, f"duplicate stop for block {idx}"
            block_stops.add(idx)
        elif t == "message_delta":
            assert not seen_message_delta, "duplicate message_delta"
            seen_message_delta = True
        elif t == "message_stop":
            assert seen_message_delta, "message_stop before message_delta"
            assert not seen_message_stop, "duplicate message_stop"
            seen_message_stop = True

    assert seen_message_start, "no message_start"
    assert seen_message_delta, "no message_delta"
    assert seen_message_stop, "no message_stop"


@pytest.mark.parametrize(
    "fixture,provider_cls,model",
    [
        (
            "tests/fixtures/ollama/stream.json",
            OllamaProvider,
            "qwen3:4b",
        ),
        (
            "tests/fixtures/vllm/stream.json",
            VLLMProvider,
            "gpt-oss-20b",
        ),
        (
            "tests/fixtures/openai_compat/stream.json",
            OpenRouterProvider,
            "mistralai/mistral-nemo",
        ),
    ],
)
def test_text_delta_concatenation(fixture, provider_cls, model):
    """Concatenated text_delta text contains 'pong'."""
    encoder = AnthropicStreamEncoder(model)
    provider = provider_cls(model=model, base_url="http://test")
    chunks = _parse_events(encoder, provider, fixture)
    text = ""
    for chunk in chunks:
        if chunk["type"] == "content_block_delta" and chunk["delta"]["type"] == "text_delta":
            text += chunk["delta"]["text"]
    assert "pong" in text.lower()


def test_vllm_block_order():
    """vLLM chat stream encodes as thinking block then text block."""
    encoder = AnthropicStreamEncoder("gpt-oss-20b")
    provider = VLLMProvider(model="gpt-oss-20b", base_url="http://test")
    chunks = _parse_events(encoder, provider, "tests/fixtures/vllm/stream.json")
    # Collect block_start events
    starts = [c for c in chunks if c["type"] == "content_block_start"]
    assert len(starts) >= 2, "expected at least thinking and text blocks"
    assert starts[0]["content_block"]["type"] == "thinking"
    assert starts[1]["content_block"]["type"] == "text"


def test_vllm_matches_reference_block_order():
    """vLLM chat stream has same block-type order as vLLM Messages reference."""
    encoder = AnthropicStreamEncoder("gpt-oss-20b")
    provider = VLLMProvider(model="gpt-oss-20b", base_url="http://test")
    chat_chunks = _parse_events(encoder, provider, "tests/fixtures/vllm/stream.json")
    chat_types = [
        c["content_block"]["type"] for c in chat_chunks if c["type"] == "content_block_start"
    ]

    # Reference: native Messages stream
    with open("tests/fixtures/vllm_messages/stream.json") as f:
        reference = json.load(f)
    ref_events = []
    for line in reference["response"]["lines"]:
        if line.startswith("data:"):
            data = line[len("data:") :].strip()
            if data:
                ref_events.append(json.loads(data))
    ref_types = [
        e["content_block"]["type"] for e in ref_events if e["type"] == "content_block_start"
    ]

    assert chat_types == ref_types


def test_tool_call():
    """Tool call encodes as tool_use block with input_json_delta."""
    encoder = AnthropicStreamEncoder("test-model")
    call = ToolCall(id="call_1", name="get_weather", input={})
    events = [
        MessageStart(id="msg-1", model="test-model"),
        BlockStart(index=0, block=call),
        BlockDelta(index=0, delta=ToolInputDelta(partial_json='{"location"')),
        BlockDelta(index=0, delta=ToolInputDelta(partial_json=': "Paris"}')),
        BlockStop(index=0),
        MessageDelta(stop_reason="tool_use", usage=None),
        MessageStop(),
    ]
    chunks = []
    for event in events:
        chunks.extend(encoder.encode(event))
    validate_anthropic_stream(chunks)

    # content_block_start should have tool_use with empty input
    start = next(c for c in chunks if c["type"] == "content_block_start")
    assert start["content_block"]["type"] == "tool_use"
    assert start["content_block"]["name"] == "get_weather"
    assert start["content_block"]["input"] == {}

    # Deltas should have input_json_delta
    deltas = [c for c in chunks if c["type"] == "content_block_delta"]
    assert len(deltas) == 2
    assert deltas[0]["delta"]["type"] == "input_json_delta"
    assert deltas[1]["delta"]["type"] == "input_json_delta"
    partial_json = deltas[0]["delta"]["partial_json"] + deltas[1]["delta"]["partial_json"]
    assert partial_json == '{"location": "Paris"}'

    # message_delta should have tool_use stop_reason
    msg_delta = next(c for c in chunks if c["type"] == "message_delta")
    assert msg_delta["delta"]["stop_reason"] == "tool_use"


@pytest.mark.parametrize(
    "internal,wire",
    [
        ("end_turn", "end_turn"),
        ("max_tokens", "max_tokens"),
        ("stop_sequence", "stop_sequence"),
        ("tool_use", "tool_use"),
        ("context_window_exceeded", "model_context_window_exceeded"),
    ],
)
def test_stop_reason_mapping(internal, wire):
    """Stop reasons map correctly from internal to wire names."""
    encoder = AnthropicStreamEncoder("test-model")
    events = [
        MessageStart(id="msg-1", model="test-model"),
        BlockStart(index=0, block=Text(text="")),
        BlockDelta(index=0, delta=TextDelta(text="test")),
        BlockStop(index=0),
        MessageDelta(stop_reason=internal, usage=None),
        MessageStop(),
    ]
    chunks = []
    for event in events:
        chunks.extend(encoder.encode(event))
    msg_delta = next(c for c in chunks if c["type"] == "message_delta")
    assert msg_delta["delta"]["stop_reason"] == wire


def test_usage_present():
    """Usage fields present when known."""
    encoder = AnthropicStreamEncoder("test-model")
    start_usage = Usage(input_tokens=10, cache_read_tokens=2, cache_write_tokens=3)
    delta_usage = Usage(output_tokens=20, input_tokens=10)
    events = [
        MessageStart(id="msg-1", model="test-model", usage=start_usage),
        MessageDelta(stop_reason="end_turn", usage=delta_usage),
        MessageStop(),
    ]
    chunks = []
    for event in events:
        chunks.extend(encoder.encode(event))

    # message_start should have input usage
    msg_start = next(c for c in chunks if c["type"] == "message_start")
    assert msg_start["message"]["usage"]["input_tokens"] == 10
    assert msg_start["message"]["usage"]["output_tokens"] == 0
    assert msg_start["message"]["usage"]["cache_read_input_tokens"] == 2
    assert msg_start["message"]["usage"]["cache_creation_input_tokens"] == 3

    # message_delta should have output usage
    msg_delta = next(c for c in chunks if c["type"] == "message_delta")
    assert msg_delta["usage"]["output_tokens"] == 20


def test_usage_absent():
    """Usage defaults when unknown."""
    encoder = AnthropicStreamEncoder("test-model")
    events = [
        MessageStart(id="msg-1", model="test-model", usage=None),
        MessageDelta(stop_reason="end_turn", usage=None),
        MessageStop(),
    ]
    chunks = []
    for event in events:
        chunks.extend(encoder.encode(event))

    msg_start = next(c for c in chunks if c["type"] == "message_start")
    assert msg_start["message"]["usage"]["input_tokens"] == 0
    assert msg_start["message"]["usage"]["output_tokens"] == 0

    msg_delta = next(c for c in chunks if c["type"] == "message_delta")
    assert msg_delta["usage"]["output_tokens"] == 0


@pytest.mark.parametrize(
    "status,error_type",
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (429, "rate_limit_error"),
        (504, "timeout_error"),
        (529, "overloaded_error"),
        (500, "api_error"),  # default
    ],
)
def test_stream_error_status_mapping(status, error_type):
    """StreamError maps status codes to Anthropic error types."""
    encoder = AnthropicStreamEncoder("test-model")
    encoder.id = "msg-1"
    events = [StreamError(status=status, message="test error")]
    chunks = []
    for event in events:
        chunks.extend(encoder.encode(event))
    assert len(chunks) == 1
    assert chunks[0]["type"] == "error"
    assert chunks[0]["error"]["type"] == error_type
    assert chunks[0]["error"]["message"] == "test error"


def test_stream_error_explicit_type():
    """StreamError with an explicit error_type uses it if recognized."""
    encoder = AnthropicStreamEncoder("test-model")
    encoder.id = "msg-1"
    events = [StreamError(status=500, message="test", error_type="overloaded_error")]
    chunks = []
    for event in events:
        chunks.extend(encoder.encode(event))
    assert chunks[0]["error"]["type"] == "overloaded_error"


def test_stream_error_unknown_type():
    """StreamError with an unknown error_type falls back to status mapping."""
    encoder = AnthropicStreamEncoder("test-model")
    encoder.id = "msg-1"
    events = [StreamError(status=429, message="test", error_type="unknown_type")]
    chunks = []
    for event in events:
        chunks.extend(encoder.encode(event))
    assert chunks[0]["error"]["type"] == "rate_limit_error"


def test_event_before_message_start():
    """Events before message_start raise RuntimeError."""
    encoder = AnthropicStreamEncoder("test-model")
    with pytest.raises(RuntimeError, match="stream event before message_start"):
        encoder.encode(BlockStart(index=0, block=Text(text="")))


def test_error_before_message_start_is_an_error_event():
    """Ollama can fail on its first line, before any message starts."""
    from model_calling.core import StreamError
    from tests.harness.sdk import validate_anthropic_error

    out = AnthropicStreamEncoder("ollama/m").encode(StreamError(status=502, message="boom"))
    assert len(out) == 1
    validate_anthropic_error(out[0])
