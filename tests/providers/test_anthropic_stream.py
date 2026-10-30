"""Offline tests for the Anthropic Messages SSE stream parser."""

import json

import pytest

from model_calling.core import (
    BlockDelta,
    BlockStart,
    BlockStop,
    MessageDelta,
    MessageStart,
    MessageStop,
    OpaqueDelta,
    SignatureDelta,
    StreamError,
    Text,
    TextDelta,
    Thinking,
    ThinkingDelta,
    ToolInputDelta,
)
from model_calling.providers.anthropic_stream import AnthropicStreamParser
from tests.harness.replay import Exchange, fixture_path

# --- helpers ---

START = {"type": "message_start", "message": {"id": "msg", "model": "m"}}
TEXT_BLOCK = {
    "type": "content_block_start",
    "index": 0,
    "content_block": {"type": "text", "text": ""},
}
THINKING_BLOCK = {
    "type": "content_block_start",
    "index": 0,
    "content_block": {"type": "thinking", "thinking": ""},
}
TOOL_BLOCK = {
    "type": "content_block_start",
    "index": 0,
    "content_block": {"type": "tool_use", "id": "call", "name": "tool", "input": {}},
}
STOP = {"type": "content_block_stop", "index": 0}
MSG_DELTA = {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
MSG_STOP = {"type": "message_stop"}


def sse(event_type: str, obj: dict) -> list[str]:
    return [f"event: {event_type}", f"data: {json.dumps(obj)}"]


def events(*payloads, close=False) -> list:
    parser = AnthropicStreamParser("anthropic", "claude-sonnet-4")
    lines = [line for p in payloads for line in sse(p["type"], p)]
    result = [e for line in lines for e in parser.feed(line)]
    return result + (parser.close() if close else [])


def deltas(evts, kind) -> str:
    field = {
        TextDelta: "text",
        ThinkingDelta: "thinking",
        SignatureDelta: "signature",
        ToolInputDelta: "partial_json",
    }[kind]
    return "".join(
        getattr(e.delta, field)
        for e in evts
        if isinstance(e, BlockDelta) and isinstance(e.delta, kind)
    )


def starts(evts, block_type) -> list:
    return [e for e in evts if isinstance(e, BlockStart) and isinstance(e.block, block_type)]


def error(evts) -> StreamError:
    found = [e for e in evts if isinstance(e, StreamError)]
    assert found, f"no StreamError in {evts}"
    return found[0]


def assert_well_formed(evts):
    assert isinstance(evts[0], MessageStart), evts[0]
    assert [type(e) for e in evts[-2:]] == [MessageDelta, MessageStop], evts[-2:]
    open_index = None
    for event in evts[1:-2]:
        if isinstance(event, BlockStart):
            assert open_index is None, f"block {event.index} started inside {open_index}"
            open_index = event.index
        elif isinstance(event, BlockStop):
            assert event.index == open_index, event
            open_index = None
        elif isinstance(event, BlockDelta):
            assert event.index == open_index, event
    assert open_index is None, f"block {open_index} still open at message end"


# --- full stream from fixture ---


def test_vllm_messages_stream_fixture():
    exchange = Exchange.load(fixture_path("vllm_messages", "stream"))
    parser = AnthropicStreamParser("anthropic", "gpt-oss-20b")
    evts = [e for line in exchange.lines for e in parser.feed(line)]
    evts += parser.close()
    assert_well_formed(evts)
    assert evts[0].id.startswith("chatcmpl-")
    assert evts[0].model == "gpt-oss-20b"
    assert evts[0].usage.input_tokens == 74
    assert [type(s.block) for s in starts(evts, (Thinking, Text))] == [Thinking, Text]
    assert deltas(evts, ThinkingDelta)
    assert deltas(evts, TextDelta) == "pong"
    assert evts[-2].stop_reason == "end_turn"
    assert evts[-2].usage.output_tokens > 0


# --- delta types ---


@pytest.mark.parametrize(
    "block,delta_payload,delta_type,expected_text",
    [
        (TEXT_BLOCK, {"type": "text_delta", "text": "Hello"}, TextDelta, "Hello"),
        (THINKING_BLOCK, {"type": "thinking_delta", "thinking": "Hmm"}, ThinkingDelta, "Hmm"),
        (THINKING_BLOCK, {"type": "signature_delta", "signature": "sig"}, SignatureDelta, "sig"),
        (
            TOOL_BLOCK,
            {"type": "input_json_delta", "partial_json": '{"a":1}'},
            ToolInputDelta,
            '{"a":1}',
        ),
        (TEXT_BLOCK, {"type": "future_delta", "data": "x"}, OpaqueDelta, None),
    ],
)
def test_delta_types(block, delta_payload, delta_type, expected_text):
    evts = events(
        START,
        block,
        {"type": "content_block_delta", "index": 0, "delta": delta_payload},
    )
    block_deltas = [e for e in evts if isinstance(e, BlockDelta)]
    assert len(block_deltas) == 1
    assert isinstance(block_deltas[0].delta, delta_type)
    if expected_text is not None:
        field = {
            TextDelta: "text",
            ThinkingDelta: "thinking",
            SignatureDelta: "signature",
            ToolInputDelta: "partial_json",
        }[delta_type]
        assert getattr(block_deltas[0].delta, field) == expected_text
    else:
        assert block_deltas[0].delta.format == "anthropic"


# --- event handling ---


def test_event_lines_and_ping_ignored():
    evts = events({"type": "ping"}, START, {"type": "ping"}, TEXT_BLOCK)
    assert len(evts) == 2
    assert isinstance(evts[0], MessageStart)
    assert isinstance(evts[1], BlockStart)


def test_unknown_event_type_ignored():
    evts = events(START, {"type": "future_event", "data": "x"}, TEXT_BLOCK)
    assert len(evts) == 2


# --- usage ---


@pytest.mark.parametrize(
    "start_usage,delta_usage,expected_input",
    [
        (
            {"input_tokens": 100, "cache_read_input_tokens": 20, "cache_creation_input_tokens": 10},
            {"output_tokens": 50},
            130,
        ),
        (
            {"input_tokens": 100, "cache_read_input_tokens": 20},
            {"input_tokens": 150, "cache_read_input_tokens": 30, "output_tokens": 50},
            180,
        ),
        # vLLM's delta shape: input_tokens but no cache fields.
        (
            {"input_tokens": 10, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 5},
            {"input_tokens": 10, "output_tokens": 7},
            115,
        ),
    ],
)
def test_usage_accumulation(start_usage, delta_usage, expected_input):
    evts = events(
        {"type": "message_start", "message": {"id": "msg", "model": "m", "usage": start_usage}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": delta_usage},
    )
    msg_delta = [e for e in evts if isinstance(e, MessageDelta)][0]
    assert msg_delta.usage.input_tokens == expected_input


# --- stop_reason ---


@pytest.mark.parametrize(
    "wire_reason,expected",
    [
        ("end_turn", "end_turn"),
        ("max_tokens", "max_tokens"),
        ("model_context_window_exceeded", "context_window_exceeded"),
        ("future_reason", None),
    ],
)
def test_stop_reason_mapping(wire_reason, expected):
    evts = events(START, {"type": "message_delta", "delta": {"stop_reason": wire_reason}})
    msg_delta = [e for e in evts if isinstance(e, MessageDelta)][0]
    assert msg_delta.stop_reason == expected


@pytest.mark.parametrize(
    "has_tool,stop_reason,expected",
    [
        (False, "end_turn", "end_turn"),
        (True, "end_turn", "tool_use"),
        (True, "max_tokens", "max_tokens"),
    ],
)
def test_stop_reason_tool_use_override(has_tool, stop_reason, expected):
    payloads = [START]
    if has_tool:
        payloads.append(TOOL_BLOCK)
    payloads.append({"type": "message_delta", "delta": {"stop_reason": stop_reason}})
    evts = events(*payloads)
    msg_delta = [e for e in evts if isinstance(e, MessageDelta)][0]
    assert msg_delta.stop_reason == expected


# --- errors ---


@pytest.mark.parametrize(
    "lines,close,message_fragment",
    [
        (["data: not json"], False, "non-JSON"),
        (["data: [1, 2]"], False, "isn't an object"),
        ([f"data: {json.dumps({'type': 'error', 'error': {'message': 'fail'}})}"], False, "fail"),
        (sse("content_block_start", TEXT_BLOCK), False, "before message_start"),
        (sse("message_start", START) + sse("message_start", START), False, "second"),
        (
            sse("message_start", START) + sse("content_block_start", TEXT_BLOCK),
            True,
            "before message_stop",
        ),
    ],
)
def test_stream_errors(lines, close, message_fragment):
    parser = AnthropicStreamParser("anthropic", "m")
    evts = [e for line in lines for e in parser.feed(line)]
    evts += parser.close() if close else []
    err = error(evts)
    assert message_fragment in err.message
    assert err.status == 502


def test_error_event_with_type():
    evts = events({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
    err = error(evts)
    assert err.error_type == "overloaded_error"
    assert "Overloaded" in err.message


def test_close_after_message_stop_returns_empty():
    parser = AnthropicStreamParser("anthropic", "m")
    for line in sse("message_start", START) + sse("message_stop", MSG_STOP):
        parser.feed(line)
    assert parser.close() == []


def test_lines_after_message_stop_ignored():
    evts = events(START, TEXT_BLOCK, STOP, MSG_DELTA, MSG_STOP)
    assert len(evts) == 5
    assert isinstance(evts[-1], MessageStop)


# --- message_start with missing fields ---


@pytest.mark.parametrize(
    "message,expected_id,expected_model,expected_usage",
    [
        ({"model": "m"}, "gen-", "m", None),
        ({"id": "msg"}, "msg", "claude-sonnet-4", None),
        ({"id": "msg", "model": "m", "usage": {"input_tokens": 10}}, "msg", "m", 10),
    ],
)
def test_message_start_fields(message, expected_id, expected_model, expected_usage):
    evts = events({"type": "message_start", "message": message})
    assert evts[0].id.startswith(expected_id)
    assert evts[0].model == expected_model
    if expected_usage:
        assert evts[0].usage.input_tokens == expected_usage
    else:
        assert evts[0].usage is None


def test_message_delta_stop_sequence():
    evts = events(
        START,
        {
            "type": "message_delta",
            "delta": {"stop_reason": "stop_sequence", "stop_sequence": "STOP"},
        },
    )
    msg_delta = [e for e in evts if isinstance(e, MessageDelta)][0]
    assert msg_delta.stop_sequence == "STOP"


def test_message_delta_stop_details_and_container():
    """stop_details and container pass through from the wire (Phase 12 item 2)."""
    stop_details = {"context_window_exceeded": {"input_tokens": 5000}}
    container = {"type": "project", "id": "proj_123"}
    evts = events(
        START,
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": "end_turn",
                "stop_details": stop_details,
                "container": container,
            },
        },
    )
    msg_delta = [e for e in evts if isinstance(e, MessageDelta)][0]
    assert msg_delta.stop_details == stop_details
    assert msg_delta.container == container
