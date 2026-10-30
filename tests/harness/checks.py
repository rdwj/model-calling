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
EXPECTED_PARAMETERLESS_TOOL = ("get_time", {})
EXPECTED_PARALLEL_TOOLS = [
    ("get_weather", {"location": "Paris"}),
    ("get_weather", {"location": "Tokyo"}),
]
EXPECTED_COLOR = "red"
EXPECTED_PRODUCT = "391"
EXPECTED_CAPITAL = {"city": "Paris", "country": "France"}


def _capital(text: str):
    """The answer to the json_schema scenario: the whole text is the JSON object."""
    assert json.loads(text) == EXPECTED_CAPITAL, text


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


def completion_reasoning(resp, model: str):
    _completion_text(resp, model, EXPECTED_PRODUCT)
    assert (resp.choices[0].message.reasoning or "").strip(), "no reasoning returned"


def completion_thinking_unsupported(resp, model: str):
    """thinking_unsupported uses THINKING_PROMPT and expects EXPECTED_PRODUCT."""
    _completion_text(resp, model, EXPECTED_PRODUCT)


def completion_json_schema(resp, model: str):
    validate_chat_completion(resp.to_dict())
    choice = resp.choices[0]
    assert choice.finish_reason == "stop"
    _capital(choice.message.content)


def _completion_calls(resp) -> list:
    sdk = validate_chat_completion(resp.to_dict())
    assert sdk.choices[0].message.function_call is None, "legacy function_call emitted"
    choice = resp.choices[0]
    assert choice.message.tool_calls, f"no tool_calls in {choice.message}"
    assert choice.finish_reason == "tool_calls"
    return [(c.function.name, json.loads(c.function.arguments)) for c in choice.message.tool_calls]


def completion_tools(resp, model: str):
    assert _completion_calls(resp)[0] == EXPECTED_TOOL


def completion_parameterless_tool(resp, model: str):
    assert _completion_calls(resp)[0] == EXPECTED_PARAMETERLESS_TOOL


def completion_parallel_tools(resp, model: str):
    assert sorted(_completion_calls(resp), key=str) == EXPECTED_PARALLEL_TOOLS


def completion_parallel_tools_off(resp, model: str):
    calls = _completion_calls(resp)
    assert len(calls) == 1 and calls[0] in EXPECTED_PARALLEL_TOOLS, calls


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


def completion_stream_thinking_unsupported(chunks, model: str):
    """Streaming thinking_unsupported expects EXPECTED_PRODUCT."""
    validate_chat_completion_chunks([c.to_dict() for c in chunks])
    finish = [
        c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason
    ]
    assert finish == ["stop"], finish
    assert len({c.id for c in chunks}) == 1, "chunks of one stream must share an id"
    assert {c.model for c in chunks} == {model}
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert EXPECTED_PRODUCT in text.lower()


def completion_stream_tools(chunks, model: str):
    validate_chat_completion_chunks([c.to_dict() for c in chunks])
    finish = [
        c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason
    ]
    assert finish == ["tool_calls"], finish
    assert {c.model for c in chunks} == {model}
    calls: dict = {}
    for c in chunks:
        for d in (c.choices[0].delta.tool_calls or []) if c.choices else []:
            name, args = calls.get(d.index, ("", ""))
            fn = d.function
            calls[d.index] = (name + (fn.name or ""), args + (fn.arguments or ""))
    assert calls, "no tool_calls deltas"
    name, args = calls[min(calls)]
    assert (name, json.loads(args)) == EXPECTED_TOOL


def completion_stream_usage(chunks, model: str):
    _completion_stream(chunks, model)
    usage = [c.usage for c in chunks if c.usage is not None]
    assert len(usage) == 1 and chunks[-1].usage is not None, "usage must come last, once"
    assert not chunks[-1].choices, "the usage chunk carries no choices"
    assert usage[0].prompt_tokens > 0 and usage[0].completion_tokens > 0


def completion_assembled_text(resp, model: str):
    """A completion assembled from a stream that didn't ask for usage."""
    validate_chat_completion(resp.to_dict())
    assert EXPECTED_TEXT in (resp.choices[0].message.content or "").lower()
    assert resp.choices[0].finish_reason == "stop"
    assert resp.model == model


def completion_assembled_thinking_unsupported(resp, model: str):
    """stream_thinking_unsupported assembled, expects EXPECTED_PRODUCT."""
    validate_chat_completion(resp.to_dict())
    assert EXPECTED_PRODUCT in (resp.choices[0].message.content or "").lower()
    assert resp.choices[0].finish_reason == "stop"
    assert resp.model == model


COMPLETION_CHECKS = {
    "text": completion_text,
    "system_blocks": completion_text,
    "tools": completion_tools,
    "parallel_tools": completion_parallel_tools,
    "tool_error": completion_tool_error,
    "image": completion_image,
    "json_schema": completion_json_schema,
    "stream": completion_stream,
    "stream_tools": completion_stream_tools,
    "stream_usage": completion_stream_usage,
    "reasoning_effort": completion_reasoning,
    "tool_required": completion_tools,
    "tool_named": completion_tools,
    "tool_strict": completion_tools,
    "tool_parameterless": completion_parameterless_tool,
    "parallel_tools_off": completion_parallel_tools_off,
    "max_completion_tokens": completion_text,
    "thinking_unsupported": completion_thinking_unsupported,
    "stream_thinking_unsupported": completion_stream_thinking_unsupported,
}


# What the SDK's stream state must assemble from each stream scenario.
FINAL_COMPLETION_CHECKS = {
    "stream": completion_assembled_text,
    "stream_tools": completion_tools,
    "stream_usage": completion_text,
    "stream_thinking_unsupported": completion_assembled_thinking_unsupported,
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


def message_thinking_unsupported(msg, model: str):
    """thinking_unsupported uses THINKING_PROMPT but doesn't return thinking blocks."""
    _message_text(msg, model, EXPECTED_PRODUCT)


def message_json_schema(msg, model: str):
    validate_anthropic_message(msg.to_dict())
    assert msg.stop_reason == "end_turn"
    _capital("".join(b.text for b in msg.content if b.type == "text"))


def _message_calls(msg) -> list:
    validate_anthropic_message(msg.to_dict())
    calls = [(b.name, b.input) for b in msg.content if b.type == "tool_use"]
    assert calls, f"no tool_use in {msg.content}"
    assert msg.stop_reason == "tool_use"
    return calls


def message_tools(msg, model: str):
    assert _message_calls(msg)[0] == EXPECTED_TOOL


def message_parameterless_tool(msg, model: str):
    assert _message_calls(msg)[0] == EXPECTED_PARAMETERLESS_TOOL


def message_parallel_tools(msg, model: str):
    assert sorted(_message_calls(msg), key=str) == EXPECTED_PARALLEL_TOOLS


def message_parallel_tools_off(msg, model: str):
    calls = _message_calls(msg)
    assert len(calls) == 1 and calls[0] in EXPECTED_PARALLEL_TOOLS, calls


def message_tool_error(msg, model: str):
    """Told the call failed, the model either explains or retries."""
    validate_anthropic_message(msg.to_dict())
    if msg.stop_reason == "tool_use":
        assert any(b.type == "tool_use" for b in msg.content)
    else:
        assert msg.stop_reason == "end_turn"
        assert "".join(b.text for b in msg.content if b.type == "text").strip(), msg.content


def _message_stream_structure(events, model: str):
    """A well-formed event sequence: start, balanced blocks, delta, stop."""
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


def _message_stream(events, model: str):
    _message_stream_structure(events, model)
    text = "".join(
        e.delta.text
        for e in events
        if e.type == "content_block_delta" and e.delta.type == "text_delta"
    )
    assert EXPECTED_TEXT in text.lower()
    assert events[-2].delta.stop_reason == "end_turn"


def message_stream(events, model: str):
    _message_stream(events, model)


def message_stream_thinking_unsupported(events, model: str):
    """Streaming thinking_unsupported expects EXPECTED_PRODUCT."""
    _message_stream_structure(events, model)
    text = "".join(
        e.delta.text
        for e in events
        if e.type == "content_block_delta" and e.delta.type == "text_delta"
    )
    assert EXPECTED_PRODUCT in text.lower()
    assert events[-2].delta.stop_reason == "end_turn"


def message_stream_tools(events, model: str):
    _message_stream_structure(events, model)
    starts = {
        e.index: e.content_block
        for e in events
        if e.type == "content_block_start" and e.content_block.type == "tool_use"
    }
    assert starts, "no tool_use block"
    index = min(starts)
    partial = "".join(
        e.delta.partial_json for e in events if e.type == "content_block_delta" and e.index == index
    )
    assert (starts[index].name, json.loads(partial)) == EXPECTED_TOOL
    assert events[-2].delta.stop_reason == "tool_use"


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
    "json_schema": message_json_schema,
    "stream": message_stream,
    "stream_tools": message_stream_tools,
    "stream_usage": message_stream_usage,
    "reasoning_effort": message_thinking,
    "tool_required": message_tools,
    "tool_named": message_tools,
    "tool_strict": message_tools,
    "tool_parameterless": message_parameterless_tool,
    "parallel_tools_off": message_parallel_tools_off,
    "thinking_unsupported": message_thinking_unsupported,
    "stream_thinking_unsupported": message_stream_thinking_unsupported,
}

# What the SDK's stream helper must assemble from each stream scenario.
FINAL_MESSAGE_CHECKS = {
    "stream": message_text,
    "stream_tools": message_tools,
    "stream_usage": message_text,
    "stream_thinking_unsupported": message_thinking_unsupported,
}
