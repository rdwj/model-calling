"""Internal stream events to Chat Completions chunks."""

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
    Thinking,
    ThinkingDelta,
    ToolCall,
    ToolInputDelta,
    Usage,
)
from model_calling.formats.chat_completions_stream import ChatCompletionsStreamEncoder
from tests.harness.sdk import validate_chat_completion_chunks


def test_stream_encoder_basic_text():
    encoder = ChatCompletionsStreamEncoder("gpt-4", include_usage=False)
    events = [
        MessageStart(id="chatcmpl-123", model="gpt-4"),
        BlockStart(index=0, block=Text(text="")),
        BlockDelta(index=0, delta=TextDelta(text="Hello")),
        BlockDelta(index=0, delta=TextDelta(text=" world")),
        BlockStop(index=0),
        MessageDelta(stop_reason="end_turn", usage=None),
        MessageStop(),
    ]
    chunks = []
    for event in events:
        chunks.extend(encoder.encode(event))
    validated = validate_chat_completion_chunks(chunks)
    assert all(c.id == "chatcmpl-123" for c in validated)
    text = "".join(c.choices[0].delta.content or "" for c in validated if c.choices)
    assert text == "Hello world"


def test_stream_encoder_thinking():
    encoder = ChatCompletionsStreamEncoder("gpt-4", include_usage=False)
    events = [
        MessageStart(id="chatcmpl-123", model="gpt-4"),
        BlockStart(index=0, block=Thinking(thinking="")),
        BlockDelta(index=0, delta=ThinkingDelta(thinking="Thinking...")),
        BlockStop(index=0),
        BlockStart(index=1, block=Text(text="")),
        BlockDelta(index=1, delta=TextDelta(text="Answer")),
        BlockStop(index=1),
        MessageDelta(stop_reason="end_turn", usage=None),
        MessageStop(),
    ]
    chunks = []
    for event in events:
        chunks.extend(encoder.encode(event))
    validated = validate_chat_completion_chunks(chunks)
    # reasoning is in the delta dict but not a formal SDK field
    reasoning = "".join(
        chunk.model_dump()["choices"][0]["delta"].get("reasoning", "")
        for chunk in validated
        if chunk.choices
    )
    content = "".join(
        c.choices[0].delta.content or "" for c in validated if c.choices and c.choices[0].delta
    )
    assert reasoning == "Thinking..."
    assert content == "Answer"


def test_stream_encoder_tool_call():
    encoder = ChatCompletionsStreamEncoder("gpt-4", include_usage=False)
    call = ToolCall(id="call_1", name="get_weather", input={})
    events = [
        MessageStart(id="chatcmpl-123", model="gpt-4"),
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
    validated = validate_chat_completion_chunks(chunks)
    # Tool call index is 0 for the first tool call
    tool_chunks = [c for c in validated if c.choices and c.choices[0].delta.tool_calls is not None]
    assert len(tool_chunks) == 3  # BlockStart + 2 deltas
    assert tool_chunks[0].choices[0].delta.tool_calls[0].index == 0


def test_stream_encoder_include_usage():
    encoder = ChatCompletionsStreamEncoder("gpt-4", include_usage=True)
    usage = Usage(input_tokens=10, output_tokens=20)
    events = [
        MessageStart(id="chatcmpl-123", model="gpt-4"),
        MessageDelta(stop_reason="end_turn", usage=usage),
        MessageStop(),
    ]
    chunks = []
    for event in events:
        chunks.extend(encoder.encode(event))
    validated = validate_chat_completion_chunks(chunks)
    # Final chunk has usage and empty choices
    assert validated[-1].choices == []
    assert validated[-1].usage.prompt_tokens == 10
    assert validated[-1].usage.completion_tokens == 20


def test_stream_encoder_error():
    encoder = ChatCompletionsStreamEncoder("gpt-4", include_usage=False)
    events = [
        MessageStart(id="chatcmpl-123", model="gpt-4"),
        StreamError(status=500, message="Backend failed"),
    ]
    chunks = []
    for event in events:
        chunks.extend(encoder.encode(event))
    # StreamError produces an error body
    assert len(chunks) == 2
    assert "error" in chunks[1]
    assert chunks[1]["error"]["message"] == "Backend failed"
