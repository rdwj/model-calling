"""Tests for Anthropic Messages types."""

import pytest
from pydantic import ValidationError

from model_calling.formats.anthropic import response_from_internal
from model_calling.types.messages import (
    ContentBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    UnknownContentBlock,
    UnknownDelta,
    parse_stream_event,
)
from tests.harness.replay import Exchange, fixture_path
from tests.harness.sdk import parse_sse, validate_anthropic_message, validate_anthropic_stream


def load_fixture(scenario: str) -> Exchange:
    return Exchange.load(fixture_path("vllm_messages", scenario))


def test_parse_text_message():
    """Parse a text response with thinking block."""
    ex = load_fixture("text")
    body = ex.response_json

    msg = Message.model_validate(body)
    assert msg.id == "chatcmpl-83e77d68a01d286d"
    assert msg.type == "message"
    assert msg.role == "assistant"
    assert msg.model == "gpt-oss-20b"
    assert msg.stop_reason == "end_turn"
    assert len(msg.content) == 2

    # Check thinking block
    thinking = msg.content[0]
    assert isinstance(thinking, ThinkingBlock)
    assert thinking.type == "thinking"
    assert thinking.signature == "8e2c48cd160b4c21b55f9bf04baf5de5"
    assert "Reply with exactly one word: pong" in thinking.thinking

    # Check text block
    text = msg.content[1]
    assert isinstance(text, TextBlock)
    assert text.type == "text"
    assert text.text == "pong"

    # Round-trip
    assert msg.to_dict() == body
    validate_anthropic_message(msg.to_dict())


def test_parse_tools_message():
    """Parse a tool use response."""
    ex = load_fixture("tools")
    body = ex.response_json

    msg = Message.model_validate(body)
    assert msg.stop_reason == "tool_use"
    assert len(msg.content) == 2

    # Check tool use block
    tool_use = msg.content[1]
    assert isinstance(tool_use, ToolUseBlock)
    assert tool_use.type == "tool_use"
    assert tool_use.id == "chatcmpl-tool-949af81aa3b58b84"
    assert tool_use.name == "get_weather"
    assert tool_use.input == {"location": "Paris"}

    # Round-trip
    assert msg.to_dict() == body
    validate_anthropic_message(msg.to_dict())


def test_parse_stream_events():
    """Parse all events in a streaming response."""
    ex = load_fixture("stream")
    events_raw = parse_sse(ex.body_bytes().decode())

    # Fill missing signature in content_block_start per fixtures README
    for event in events_raw:
        if (
            isinstance(event, dict)
            and event.get("type") == "content_block_start"
            and event.get("content_block", {}).get("type") == "thinking"
            and "signature" not in event["content_block"]
        ):
            event["content_block"]["signature"] = ""

    # Parse all events
    events = [parse_stream_event(e) for e in events_raw]
    assert len(events) > 0

    # Verify event sequence
    types = [e.type for e in events]
    assert types[0] == "message_start"
    assert types[-1] == "message_stop"
    assert types[-2] == "message_delta"

    # Check message_start
    msg_start = events[0]
    assert msg_start.message.id.startswith("chatcmpl-")
    assert msg_start.message.role == "assistant"
    assert msg_start.message.content == []
    assert msg_start.message.stop_reason is None

    # Check content block events
    block_starts = [e for e in events if e.type == "content_block_start"]
    assert len(block_starts) == 2  # thinking + text

    # Check deltas
    deltas = [e for e in events if e.type == "content_block_delta"]
    delta_types = {d.delta.type for d in deltas}
    assert delta_types == {"thinking_delta", "signature_delta", "text_delta"}

    # Verify text reconstruction
    text_deltas = [d.delta.text for d in deltas if d.delta.type == "text_delta"]
    assert "".join(text_deltas) == "pong"

    # Check message_delta
    msg_delta = events[-2]
    assert msg_delta.delta.stop_reason == "end_turn"
    assert msg_delta.usage.output_tokens > 0

    # Round-trip all events
    for i, event in enumerate(events):
        roundtrip = event.to_dict()
        assert roundtrip == events_raw[i]

    # Validate against SDK
    validate_anthropic_stream([e.to_dict() for e in events])


def test_response_from_internal_unsigned_thinking():
    """Bodies from response_from_internal parse and round-trip."""
    from model_calling.core.blocks import Text, Thinking
    from model_calling.core.response import Response, Usage as InternalUsage

    # Create internal response with unsigned thinking
    internal = Response(
        id="msg_test123",
        model="test-model",
        content=[
            Thinking(thinking="Let me think...", signature=""),
            Text(text="Hello"),
        ],
        stop_reason="end_turn",
        usage=InternalUsage(input_tokens=10, output_tokens=20),
    )

    # Convert to wire format
    body = response_from_internal(internal)

    # Parse with our types
    msg = Message.model_validate(body)
    assert len(msg.content) == 2
    assert msg.content[0].thinking == "Let me think..."
    assert msg.content[0].signature == ""  # Unsigned
    assert msg.content[1].text == "Hello"

    # Round-trip
    assert msg.to_dict() == body


def test_response_from_internal_tool_call():
    """Parse response with tool call."""
    from model_calling.core.blocks import ToolCall
    from model_calling.core.response import Response, Usage as InternalUsage

    internal = Response(
        id="msg_test456",
        model="test-model",
        content=[
            ToolCall(
                id="call_abc",
                name="get_weather",
                input={"location": "NYC"},
            )
        ],
        stop_reason="tool_use",
        usage=InternalUsage(input_tokens=15, output_tokens=25),
    )

    body = response_from_internal(internal)
    msg = Message.model_validate(body)

    assert msg.stop_reason == "tool_use"
    tool = msg.content[0]
    assert tool.name == "get_weather"
    assert tool.input == {"location": "NYC"}

    # Round-trip
    assert msg.to_dict() == body


def test_response_from_internal_cache_usage():
    """Parse response with cache usage fields."""
    from model_calling.core.blocks import Text
    from model_calling.core.response import Response, Usage as InternalUsage

    internal = Response(
        id="msg_test789",
        model="test-model",
        content=[Text(text="Cached response")],
        stop_reason="end_turn",
        usage=InternalUsage(
            input_tokens=5,
            output_tokens=10,
            cache_write_tokens=100,
            cache_read_tokens=50,
        ),
    )

    body = response_from_internal(internal)
    msg = Message.model_validate(body)

    assert msg.usage.cache_creation_input_tokens == 100
    assert msg.usage.cache_read_input_tokens == 50

    # Round-trip
    assert msg.to_dict() == body
    validate_anthropic_message(msg.to_dict())


def test_thinking_block_requires_signature():
    """ThinkingBlock without signature fails validation."""
    with pytest.raises(ValidationError) as exc_info:
        ThinkingBlock.model_validate({"type": "thinking", "thinking": "Hello"})

    errors = exc_info.value.errors()
    assert any("signature" in str(e) for e in errors)


def test_unknown_block_type_survives():
    """Unknown content block types parse with fallback."""
    block_data = {
        "type": "server_tool_use",
        "id": "x",
        "name": "web_search",
        "input": {"query": "anthropic"},
    }

    # Parse as ContentBlock union
    from pydantic import TypeAdapter

    adapter = TypeAdapter(ContentBlock)
    block = adapter.validate_python(block_data)

    # Should be UnknownContentBlock
    assert isinstance(block, UnknownContentBlock)
    assert block.type == "server_tool_use"

    # Round-trip preserves fields
    roundtrip = block.to_dict()
    assert roundtrip["type"] == "server_tool_use"
    assert roundtrip["id"] == "x"
    assert roundtrip["name"] == "web_search"


def test_unknown_delta_type_survives():
    """Unknown delta types parse with fallback."""
    delta_data = {"type": "citations_delta", "citations": []}

    # Parse as Delta union
    from pydantic import TypeAdapter

    from model_calling.types.messages import Delta

    adapter = TypeAdapter(Delta)
    delta = adapter.validate_python(delta_data)

    # Should be UnknownDelta
    assert isinstance(delta, UnknownDelta)
    assert delta.type == "citations_delta"

    # Round-trip preserves fields
    roundtrip = delta.to_dict()
    assert roundtrip["type"] == "citations_delta"
    assert roundtrip["citations"] == []
