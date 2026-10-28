"""Tests for Chat Completions response types."""

import pytest
from pydantic import ValidationError

from model_calling.core import (
    BlockDelta,
    BlockStart,
    MessageDelta,
    MessageStart,
    MessageStop,
    Response,
    Text,
    TextDelta,
    Thinking,
    ThinkingDelta,
    ToolCall,
    ToolInputDelta,
    Usage,
)
from model_calling.formats.chat_completions import response_from_internal
from model_calling.formats.chat_completions_stream import ChatCompletionsStreamEncoder
from model_calling.types.chat import ChatCompletion, ChatCompletionChunk
from tests.harness.sdk import validate_chat_completion, validate_chat_completion_chunks

# Representative internal responses for testing


def _text_response():
    return Response(
        id="chatcmpl-123",
        model="gpt-4o",
        content=[Text(text="Hello, world!")],
        stop_reason="end_turn",
    )


def _tool_call_response():
    return Response(
        id="chatcmpl-456",
        model="gpt-4o",
        content=[
            ToolCall(id="call_1", name="get_weather", input={"location": "SF"}),
            ToolCall(id="call_2", name="get_time", input={"timezone": "UTC"}),
        ],
        stop_reason="tool_use",
    )


def _thinking_response():
    return Response(
        id="chatcmpl-789",
        model="o1",
        content=[
            Thinking(thinking="Let me think about this..."),
            Text(text="The answer is 42."),
        ],
        stop_reason="end_turn",
        usage=Usage(input_tokens=100, output_tokens=50, reasoning_tokens=30),
    )


def _usage_with_cache():
    return Response(
        id="chatcmpl-cache",
        model="gpt-4o",
        content=[Text(text="Cached response")],
        stop_reason="end_turn",
        usage=Usage(
            input_tokens=1000,
            output_tokens=200,
            cache_read_tokens=800,
            cache_write_tokens=200,
            reasoning_tokens=None,
        ),
    )


def _no_usage_response():
    return Response(
        id="chatcmpl-no-usage",
        model="gpt-4o",
        content=[Text(text="No usage")],
        stop_reason="end_turn",
        usage=None,
    )


# Streaming event sequences


def _text_stream_events():
    """Text-only streaming with usage at the end."""
    return [
        MessageStart(id="chatcmpl-stream-1", model="gpt-4o", usage=None),
        BlockStart(index=0, block=Text(text="")),
        BlockDelta(index=0, delta=TextDelta(text="Hello")),
        BlockDelta(index=0, delta=TextDelta(text=", world!")),
        MessageDelta(stop_reason="end_turn", usage=Usage(input_tokens=10, output_tokens=5)),
        MessageStop(),
    ]


def _tool_call_stream_events():
    """Two parallel tool calls."""
    return [
        MessageStart(id="chatcmpl-stream-2", model="gpt-4o", usage=None),
        BlockStart(index=0, block=ToolCall(id="call_1", name="get_weather", input={})),
        BlockDelta(index=0, delta=ToolInputDelta(partial_json='{"location"')),
        BlockDelta(index=0, delta=ToolInputDelta(partial_json=': "SF"}')),
        BlockStart(index=1, block=ToolCall(id="call_2", name="get_time", input={})),
        BlockDelta(index=1, delta=ToolInputDelta(partial_json='{"timezone"')),
        BlockDelta(index=1, delta=ToolInputDelta(partial_json=': "UTC"}')),
        MessageDelta(stop_reason="tool_use", usage=Usage(input_tokens=20, output_tokens=15)),
        MessageStop(),
    ]


def _thinking_stream_events():
    """Thinking block followed by text."""
    return [
        MessageStart(id="chatcmpl-stream-3", model="o1", usage=None),
        BlockStart(index=0, block=Thinking(thinking="")),
        BlockDelta(index=0, delta=ThinkingDelta(thinking="Reasoning")),
        BlockDelta(index=0, delta=ThinkingDelta(thinking="...")),
        BlockStart(index=1, block=Text(text="")),
        BlockDelta(index=1, delta=TextDelta(text="Done.")),
        MessageDelta(
            stop_reason="end_turn",
            usage=Usage(input_tokens=50, output_tokens=30, reasoning_tokens=20),
        ),
        MessageStop(),
    ]


# Tests


def test_chat_completion_text_only():
    resp = _text_response()
    wire = response_from_internal(resp, "gpt-4o")

    # Parse with our types
    parsed = ChatCompletion.model_validate(wire)

    # Lossless round-trip
    assert parsed.to_dict() == wire

    # SDK validation
    validate_chat_completion(parsed.to_dict())

    # Typed access
    assert parsed.id == "chatcmpl-123"
    assert parsed.model == "gpt-4o"
    assert parsed.object == "chat.completion"
    assert len(parsed.choices) == 1
    assert parsed.choices[0].index == 0
    assert parsed.choices[0].message.role == "assistant"
    assert parsed.choices[0].message.content == "Hello, world!"
    assert parsed.choices[0].finish_reason == "stop"


def test_chat_completion_tool_calls():
    resp = _tool_call_response()
    wire = response_from_internal(resp, "gpt-4o")

    parsed = ChatCompletion.model_validate(wire)
    assert parsed.to_dict() == wire
    validate_chat_completion(parsed.to_dict())

    # Two parallel tool calls
    assert len(parsed.choices[0].message.tool_calls) == 2
    assert parsed.choices[0].message.tool_calls[0].id == "call_1"
    assert parsed.choices[0].message.tool_calls[0].function.name == "get_weather"
    assert parsed.choices[0].message.tool_calls[0].function.arguments == '{"location": "SF"}'
    assert parsed.choices[0].message.tool_calls[1].id == "call_2"
    assert parsed.choices[0].message.tool_calls[1].function.name == "get_time"
    assert parsed.choices[0].finish_reason == "tool_calls"


def test_chat_completion_thinking():
    resp = _thinking_response()
    wire = response_from_internal(resp, "o1")

    parsed = ChatCompletion.model_validate(wire)
    assert parsed.to_dict() == wire
    validate_chat_completion(parsed.to_dict())

    # Reasoning field (non-standard)
    assert parsed.choices[0].message.reasoning == "Let me think about this..."
    assert parsed.choices[0].message.content == "The answer is 42."

    # Usage with reasoning tokens
    assert parsed.usage is not None
    assert parsed.usage.prompt_tokens == 100
    assert parsed.usage.completion_tokens == 50
    assert parsed.usage.total_tokens == 150
    assert parsed.usage.completion_tokens_details is not None
    assert parsed.usage.completion_tokens_details.reasoning_tokens == 30


def test_chat_completion_usage_with_cache():
    resp = _usage_with_cache()
    wire = response_from_internal(resp, "gpt-4o")

    parsed = ChatCompletion.model_validate(wire)
    assert parsed.to_dict() == wire
    validate_chat_completion(parsed.to_dict())

    # Cache token details
    assert parsed.usage is not None
    assert parsed.usage.prompt_tokens_details is not None
    assert parsed.usage.prompt_tokens_details.cached_tokens == 800
    assert parsed.usage.prompt_tokens_details.cache_write_tokens == 200


def test_chat_completion_no_usage():
    resp = _no_usage_response()
    wire = response_from_internal(resp, "gpt-4o")

    parsed = ChatCompletion.model_validate(wire)
    assert parsed.to_dict() == wire
    validate_chat_completion(parsed.to_dict())

    # No usage field
    assert parsed.usage is None


def test_chat_completion_chunk_text_stream_with_usage():
    events = _text_stream_events()
    encoder = ChatCompletionsStreamEncoder(model="gpt-4o", include_usage=True)
    chunks = [c for event in events for c in encoder.encode(event)]

    # Parse all chunks
    parsed_chunks = [ChatCompletionChunk.model_validate(c) for c in chunks]

    # Lossless
    assert [c.to_dict() for c in parsed_chunks] == chunks

    # SDK validation
    validate_chat_completion_chunks(chunks)

    # First chunk has role
    assert parsed_chunks[0].choices[0].delta.role == "assistant"
    assert parsed_chunks[0].choices[0].delta.content == ""

    # Text deltas
    assert parsed_chunks[1].choices[0].delta.content == "Hello"
    assert parsed_chunks[2].choices[0].delta.content == ", world!"

    # Finish reason
    assert parsed_chunks[3].choices[0].finish_reason == "stop"

    # Final chunk with usage (include_usage=True)
    assert parsed_chunks[4].choices == []
    assert parsed_chunks[4].usage is not None
    assert parsed_chunks[4].usage.prompt_tokens == 10
    assert parsed_chunks[4].usage.completion_tokens == 5


def test_chat_completion_chunk_text_stream_without_usage():
    events = _text_stream_events()
    encoder = ChatCompletionsStreamEncoder(model="gpt-4o", include_usage=False)
    chunks = [c for event in events for c in encoder.encode(event)]

    parsed_chunks = [ChatCompletionChunk.model_validate(c) for c in chunks]
    assert [c.to_dict() for c in parsed_chunks] == chunks
    validate_chat_completion_chunks(chunks)

    # No final usage chunk
    assert all(c.usage is None for c in parsed_chunks)


def test_chat_completion_chunk_tool_calls():
    events = _tool_call_stream_events()
    encoder = ChatCompletionsStreamEncoder(model="gpt-4o", include_usage=True)
    chunks = [c for event in events for c in encoder.encode(event)]

    parsed_chunks = [ChatCompletionChunk.model_validate(c) for c in chunks]
    assert [c.to_dict() for c in parsed_chunks] == chunks
    validate_chat_completion_chunks(chunks)

    # Tool call starts
    assert parsed_chunks[1].choices[0].delta.tool_calls is not None
    assert parsed_chunks[1].choices[0].delta.tool_calls[0].index == 0
    assert parsed_chunks[1].choices[0].delta.tool_calls[0].id == "call_1"

    # Tool input deltas
    assert parsed_chunks[2].choices[0].delta.tool_calls[0].function.arguments == '{"location"'
    assert parsed_chunks[3].choices[0].delta.tool_calls[0].function.arguments == ': "SF"}'

    # Second tool call
    assert parsed_chunks[4].choices[0].delta.tool_calls[0].index == 1
    assert parsed_chunks[4].choices[0].delta.tool_calls[0].id == "call_2"


def test_chat_completion_chunk_thinking():
    events = _thinking_stream_events()
    encoder = ChatCompletionsStreamEncoder(model="o1", include_usage=True)
    chunks = [c for event in events for c in encoder.encode(event)]

    parsed_chunks = [ChatCompletionChunk.model_validate(c) for c in chunks]
    assert [c.to_dict() for c in parsed_chunks] == chunks
    validate_chat_completion_chunks(chunks)

    # Reasoning deltas (non-standard)
    assert parsed_chunks[1].choices[0].delta.reasoning == "Reasoning"
    assert parsed_chunks[2].choices[0].delta.reasoning == "..."

    # Text follows
    assert parsed_chunks[3].choices[0].delta.content == "Done."

    # Usage with reasoning tokens
    assert parsed_chunks[5].usage is not None
    assert parsed_chunks[5].usage.completion_tokens_details.reasoning_tokens == 20


def test_chat_completion_missing_required_field():
    # Missing choices field
    invalid = {
        "id": "chatcmpl-bad",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o",
        # Missing: choices
    }

    with pytest.raises(ValidationError):
        ChatCompletion.model_validate(invalid)


def test_chat_completion_extra_field_survives():
    resp = _text_response()
    wire = response_from_internal(resp, "gpt-4o")

    # Add unknown field
    wire["custom_field"] = "custom_value"

    parsed = ChatCompletion.model_validate(wire)

    # Extra field survives to_dict
    assert parsed.to_dict()["custom_field"] == "custom_value"
