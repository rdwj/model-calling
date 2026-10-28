"""Offline tests for the Chat Completions SSE stream parser."""

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
    Thinking,
    ThinkingDelta,
    ToolCall,
    ToolInputDelta,
)
from model_calling.providers.openai_compat_stream import ChatCompletionsStreamParser
from tests.harness.replay import fixture_path

DONE = "data: [DONE]"


def sse(obj) -> str:
    return f"data: {json.dumps(obj)}"


def chunk(delta: dict | None = None, finish: str | None = None) -> str:
    """One SSE line carrying a single choice."""
    choice = {"index": 0, "delta": delta or {}, "finish_reason": finish}
    return sse({"id": "chatcmpl-1", "model": "m", "choices": [choice]})


def tool(index: int, id: str | None = None, name: str | None = None, args: str | None = None):
    fragment: dict = {"index": index, "function": {}}
    if id is not None:
        fragment["id"] = id
    if name is not None:
        fragment["function"]["name"] = name
    if args is not None:
        fragment["function"]["arguments"] = args
    return chunk({"tool_calls": [fragment]})


def feed_all(lines, close: bool = False) -> list:
    parser = ChatCompletionsStreamParser("vllm", "m")
    events = [e for line in lines for e in parser.feed(line)]
    return events + (parser.close() if close else [])


def deltas(events, kind) -> str:
    field = {TextDelta: "text", ThinkingDelta: "thinking", ToolInputDelta: "partial_json"}[kind]
    return "".join(
        getattr(e.delta, field)
        for e in events
        if isinstance(e, BlockDelta) and isinstance(e.delta, kind)
    )


def starts(events, block_type) -> list:
    return [e for e in events if isinstance(e, BlockStart) and isinstance(e.block, block_type)]


def error(events) -> StreamError:
    found = [e for e in events if isinstance(e, StreamError)]
    assert found, f"no StreamError in {events}"
    return found[0]


def assert_well_formed(events):
    """MessageStart first, one block open at a time, MessageDelta then MessageStop last."""
    assert isinstance(events[0], MessageStart), events[0]
    assert [type(e) for e in events[-2:]] == [MessageDelta, MessageStop], events[-2:]
    open_index = None
    for event in events[1:-2]:
        if isinstance(event, BlockStart):
            assert open_index is None, f"block {event.index} started inside {open_index}"
            open_index = event.index
        elif isinstance(event, BlockStop):
            assert event.index == open_index, event
            open_index = None
        elif isinstance(event, BlockDelta):
            assert event.index == open_index, event
    assert open_index is None, f"block {open_index} still open at message end"


def replay(backend: str) -> tuple[list, list]:
    lines = json.loads(fixture_path(backend, "stream").read_text())["response"]["lines"]
    return lines, feed_all(lines, close=True)


# --- recorded streams ---


def test_vllm_stream_fixture():
    lines, events = replay("vllm")
    assert_well_formed(events)
    first = json.loads(lines[0][len("data:") :])
    assert (events[0].id, events[0].model) == (first["id"], first["model"])
    assert [type(s.block) for s in starts(events, (Thinking, Text))] == [Thinking, Text]
    assert deltas(events, ThinkingDelta)
    assert deltas(events, TextDelta) == "pong"
    final = events[-2]
    assert final.stop_reason == "end_turn"
    assert final.usage.input_tokens > 0 and final.usage.output_tokens > 0


def test_openrouter_stream_fixture():
    _, events = replay("openai_compat")
    assert_well_formed(events)
    assert "pong" in deltas(events, TextDelta).lower()
    assert events[-2].stop_reason == "end_turn"


# --- tool calls ---


def test_tool_call_split_over_fragments():
    events = feed_all(
        [
            chunk({"role": "assistant"}),
            tool(0, id="call_1", name="get_weather"),
            tool(0, args='{"loc'),
            tool(0, args='ation": "Paris"}'),
            chunk(finish="tool_calls"),
            DONE,
        ]
    )
    assert_well_formed(events)
    (start,) = starts(events, ToolCall)
    assert (start.block.id, start.block.name, start.block.input) == ("call_1", "get_weather", {})
    assert deltas(events, ToolInputDelta) == '{"location": "Paris"}'
    assert events[-2].stop_reason == "tool_use"


def test_sequential_tool_calls_are_separate_blocks():
    events = feed_all(
        [
            chunk({"content": "Checking."}),
            tool(0, id="call_1", name="one", args="{}"),
            tool(1, id="call_2", name="two", args="{}"),
            chunk(finish="tool_calls"),
            DONE,
        ]
    )
    assert_well_formed(events)
    assert [(s.index, s.block.name) for s in starts(events, ToolCall)] == [(1, "one"), (2, "two")]


def test_tool_call_without_id_gets_one():
    events = feed_all([tool(0, name="get_weather", args="{}"), chunk(finish="tool_calls"), DONE])
    assert starts(events, ToolCall)[0].block.id.startswith("call_")


def test_finish_stop_with_tool_calls_is_tool_use():
    events = feed_all([tool(0, id="c", name="f", args="{}"), chunk(finish="stop"), DONE])
    assert events[-2].stop_reason == "tool_use"


def test_new_id_on_a_reused_index_is_a_new_call():
    events = feed_all(
        [
            tool(0, id="c1", name="f", args='{"a": 1}'),
            tool(0, id="c2", name="g", args='{"b": 2}'),
            chunk(finish="tool_calls"),
            DONE,
        ]
    )
    assert_well_formed(events)
    assert [(s.block.id, s.block.name) for s in starts(events, ToolCall)] == [
        ("c1", "f"),
        ("c2", "g"),
    ]


def test_fragment_for_closed_tool_call_is_error():
    events = feed_all(
        [
            tool(0, id="call_1", name="one", args="{}"),
            tool(1, id="call_2", name="two"),
            tool(0, args="more"),
        ]
    )
    assert "interleaved" in error(events).message


# --- framing, usage, and reasoning ---


def test_comments_and_blank_lines_are_ignored():
    lines = [chunk({"content": "Hi"}), ": OPENROUTER PROCESSING", "", chunk({"content": " there"})]
    events = feed_all(lines + [chunk(finish="stop"), DONE])
    assert_well_formed(events)
    assert deltas(events, TextDelta) == "Hi there"


def test_usage_after_finish_reaches_message_delta():
    usage = {"prompt_tokens": 10, "completion_tokens": 5}
    events = feed_all(
        [
            chunk({"content": "Hi"}),
            chunk(finish="stop"),
            sse({"id": "chatcmpl-1", "choices": [], "usage": usage}),
            DONE,
        ]
    )
    assert (events[-2].usage.input_tokens, events[-2].usage.output_tokens) == (10, 5)


def test_lines_after_done_are_ignored():
    lines = [chunk({"content": "Hi"}), chunk(finish="stop"), DONE, chunk({"content": "late"})]
    events = feed_all(lines)
    assert deltas(events, TextDelta) == "Hi"
    assert isinstance(events[-1], MessageStop)


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_reasoning_fields_become_a_thinking_block(field):
    events = feed_all(
        [
            chunk({field: "Thinking"}),
            chunk({field: " hard"}),
            chunk({"content": "Answer"}),
            chunk(finish="stop"),
            DONE,
        ]
    )
    assert_well_formed(events)
    assert [type(s.block) for s in starts(events, (Thinking, Text))] == [Thinking, Text]
    assert deltas(events, ThinkingDelta) == "Thinking hard"
    assert deltas(events, TextDelta) == "Answer"


def test_streamed_refusal_is_text_with_refusal_stop():
    events = feed_all(
        [chunk({"refusal": "I can't"}), chunk({"refusal": " help."}), chunk(finish="stop"), DONE]
    )
    assert_well_formed(events)
    assert deltas(events, TextDelta) == "I can't help."
    assert events[-2].stop_reason == "refusal"


def test_finish_without_done_still_finishes():
    events = feed_all([chunk({"content": "Hi"}), chunk(finish="stop")], close=True)
    assert_well_formed(events)
    assert events[-2].stop_reason == "end_turn"


# --- errors ---


@pytest.mark.parametrize(
    "lines, close, message",
    [
        (["data: not json"], False, "non-JSON"),
        (["data: [1, 2]"], False, "isn't an object"),
        ([sse({"error": {"message": "Model not found"}})], False, "Model not found"),
        ([chunk({"content": "Hi"})], True, "before it finished"),
        ([DONE], False, "without a response"),
    ],
    ids=["non_json", "not_object", "error_chunk", "no_done", "done_first"],
)
def test_stream_errors(lines, close, message):
    err = error(feed_all(lines, close=close))
    assert message in err.message
    assert err.status == 502
