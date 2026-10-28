"""
What each grid scenario must return, checked on the typed result.

The library, the gateway, and the SDK-client cells produce the same result
types (an `openai` ChatCompletion or chunk list, an `anthropic` Message or
stream event list), so they share these checks. Each takes the result and
the model id the client sent. Failures that raise (backend_error) are
checked where the call is made.
"""

import json

from tests.harness.sdk import (
    validate_anthropic_message,
    validate_anthropic_stream,
    validate_chat_completion,
    validate_chat_completion_chunks,
)

EXPECTED_TEXT = "pong"
EXPECTED_TOOL = ("get_weather", {"location": "Paris"})
EXPECTED_PARALLEL_TOOLS = [
    ("get_weather", {"location": "Paris"}),
    ("get_weather", {"location": "Tokyo"}),
]
EXPECTED_COLOR = "red"
EXPECTED_PRODUCT = "391"


# --- Chat Completions -----------------------------------------------------


def _completion_text(resp, model: str, expected: str):
    validate_chat_completion(resp.to_dict())
    choice = resp.choices[0]
    assert expected in (choice.message.content or "").lower(), choice.message
    assert choice.finish_reason == "stop"
    assert resp.model == model
    assert resp.usage is not None and resp.usage.total_tokens > 0


def completion_text(resp, model: str):
    _completion_text(resp, model, EXPECTED_TEXT)


def completion_image(resp, model: str):
    _completion_text(resp, model, EXPECTED_COLOR)


def _completion_calls(resp) -> list:
    sdk = validate_chat_completion(resp.to_dict())
    assert sdk.choices[0].message.function_call is None, "legacy function_call emitted"
    choice = resp.choices[0]
    assert choice.message.tool_calls, f"no tool_calls in {choice.message}"
    assert choice.finish_reason == "tool_calls"
    return [(c.function.name, json.loads(c.function.arguments)) for c in choice.message.tool_calls]


def completion_tools(resp, model: str):
    assert _completion_calls(resp)[0] == EXPECTED_TOOL


def completion_parallel_tools(resp, model: str):
    assert sorted(_completion_calls(resp), key=str) == EXPECTED_PARALLEL_TOOLS


def completion_tool_error(resp, model: str):
    """Told the call failed, the model either explains or retries."""
    validate_chat_completion(resp.to_dict())
    choice = resp.choices[0]
    if choice.finish_reason == "tool_calls":
        assert choice.message.tool_calls
    else:
        assert choice.finish_reason == "stop"
        assert (choice.message.content or "").strip(), choice.message


def _completion_stream(chunks, model: str) -> str:
    validate_chat_completion_chunks([c.to_dict() for c in chunks])
    finish = [
        c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason
    ]
    assert finish == ["stop"], finish
    assert len({c.id for c in chunks}) == 1, "chunks of one stream must share an id"
    assert {c.model for c in chunks} == {model}
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert EXPECTED_TEXT in text.lower()
    return text


def completion_stream(chunks, model: str):
    _completion_stream(chunks, model)


def completion_stream_usage(chunks, model: str):
    _completion_stream(chunks, model)
    usage = [c.usage for c in chunks if c.usage is not None]
    assert len(usage) == 1 and chunks[-1].usage is not None, "usage must come last, once"
    assert not chunks[-1].choices, "the usage chunk carries no choices"
    assert usage[0].prompt_tokens > 0 and usage[0].completion_tokens > 0


COMPLETION_CHECKS = {
    "text": completion_text,
    "system_blocks": completion_text,
    "tools": completion_tools,
    "parallel_tools": completion_parallel_tools,
    "tool_error": completion_tool_error,
    "image": completion_image,
    "stream": completion_stream,
    "stream_usage": completion_stream_usage,
}


# --- Anthropic Messages ---------------------------------------------------


def _message_text(msg, model: str, expected: str):
    validate_anthropic_message(msg.to_dict())
    text = "".join(b.text for b in msg.content if b.type == "text")
    assert expected in text.lower(), msg.content
    assert msg.stop_reason == "end_turn"
    assert msg.model == model
    assert msg.usage.output_tokens > 0


def message_text(msg, model: str):
    _message_text(msg, model, EXPECTED_TEXT)


def message_image(msg, model: str):
    _message_text(msg, model, EXPECTED_COLOR)


def message_thinking(msg, model: str):
    _message_text(msg, model, EXPECTED_PRODUCT)
    kinds = [b.type for b in msg.content]
    assert kinds[0] == "thinking", kinds
    assert msg.content[0].thinking.strip()


def _message_calls(msg) -> list:
    validate_anthropic_message(msg.to_dict())
    calls = [(b.name, b.input) for b in msg.content if b.type == "tool_use"]
    assert calls, f"no tool_use in {msg.content}"
    assert msg.stop_reason == "tool_use"
    return calls


def message_tools(msg, model: str):
    assert _message_calls(msg)[0] == EXPECTED_TOOL


def message_parallel_tools(msg, model: str):
    assert sorted(_message_calls(msg), key=str) == EXPECTED_PARALLEL_TOOLS


def message_tool_error(msg, model: str):
    """Told the call failed, the model either explains or retries."""
    validate_anthropic_message(msg.to_dict())
    if msg.stop_reason == "tool_use":
        assert any(b.type == "tool_use" for b in msg.content)
    else:
        assert msg.stop_reason == "end_turn"
        assert "".join(b.text for b in msg.content if b.type == "text").strip(), msg.content


def _message_stream(events, model: str):
    validate_anthropic_stream([e.to_dict() for e in events])
    kinds = [e.type for e in events]
    assert kinds[0] == "message_start" and kinds[-2:] == ["message_delta", "message_stop"], kinds
    assert events[0].message.model == model
    open_blocks = set()
    for e in events:
        if e.type == "content_block_start":
            assert e.index not in open_blocks
            open_blocks.add(e.index)
        elif e.type in ("content_block_delta", "content_block_stop"):
            assert e.index in open_blocks, f"{e.type} for unopened block {e.index}"
            if e.type == "content_block_stop":
                open_blocks.remove(e.index)
    assert not open_blocks, f"blocks never stopped: {open_blocks}"
    text = "".join(
        e.delta.text
        for e in events
        if e.type == "content_block_delta" and e.delta.type == "text_delta"
    )
    assert EXPECTED_TEXT in text.lower()
    assert events[-2].delta.stop_reason == "end_turn"


def message_stream(events, model: str):
    _message_stream(events, model)


def message_stream_usage(events, model: str):
    _message_stream(events, model)
    # Chat Completions backends report usage at the end, so both counts
    # arrive in message_delta rather than message_start.
    usage = events[-2].usage
    assert usage.input_tokens and usage.input_tokens > 0, usage
    assert usage.output_tokens > 0, usage


MESSAGE_CHECKS = {
    "text": message_text,
    "system_blocks": message_text,
    "tools": message_tools,
    "parallel_tools": message_parallel_tools,
    "tool_error": message_tool_error,
    "image": message_image,
    "thinking": message_thinking,
    "stream": message_stream,
    "stream_usage": message_stream_usage,
}
