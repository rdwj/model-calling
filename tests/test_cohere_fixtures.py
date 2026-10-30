"""
The hand-built Cohere v2 fixtures hold to the v2 API reference.

There is no Cohere SDK in the dependencies to validate against, so the field
sets below are copied from the reference pages, which tests/fixtures/README.md
cites: https://docs.cohere.com/reference/chat (request, response, errors),
https://docs.cohere.com/reference/chat-stream (event types), and
https://docs.cohere.com/reference/embed. The grid proves our provider sends
each request body exactly; these tests prove both halves of every exchange
use only what the reference defines.
"""

import json

import pytest

from tests.harness.replay import FIXTURES_DIR, Exchange

# --- /v2/chat request ---
CHAT_REQUEST_FIELDS = {
    "stream", "model", "messages", "tools", "documents", "citation_options",
    "response_format", "safety_mode", "max_tokens", "stop_sequences", "temperature",
    "seed", "frequency_penalty", "presence_penalty", "k", "p", "logprobs",
    "tool_choice", "thinking", "priority", "strict_tools",
}  # fmt: skip
MESSAGE_FIELDS = {
    "user": {"role", "content"},
    "assistant": {"role", "citations", "content", "tool_calls", "tool_plan"},
    "system": {"role", "content"},
    "tool": {"role", "content", "tool_call_id"},
}
# Content block type -> its fields, per role (strings are always allowed).
CONTENT_BLOCKS = {
    "user": {"text": {"type", "text"}, "image_url": {"type", "image_url"}},
    "assistant": {"text": {"type", "text"}, "thinking": {"type", "thinking"}},
    "system": {"text": {"type", "text"}},
    "tool": {"text": {"type", "text"}, "document": {"type", "document"}},
}
TOOL_CHOICES = {"REQUIRED", "NONE"}
THINKING_TYPES = {"enabled", "disabled"}
MAX_STOP_SEQUENCES = 5

# --- /v2/chat response ---
RESPONSE_FIELDS = {"id", "finish_reason", "message", "usage", "logprobs"}
FINISH_REASONS = {"COMPLETE", "STOP_SEQUENCE", "MAX_TOKENS", "TOOL_CALL", "ERROR", "TIMEOUT"}
RESPONSE_MESSAGE_FIELDS = {"role", "tool_calls", "tool_plan", "content", "citations"}
USAGE_FIELDS = {"billed_units", "tokens", "cached_tokens"}
ERROR_FIELDS = {"message", "id"}

# --- /v2/chat stream: event type -> its top-level fields, and delta.message's ---
STREAM_EVENTS = {
    "message-start": ({"type", "id", "delta"}, {"role"}),
    "content-start": ({"type", "index", "delta"}, {"content"}),
    "content-delta": ({"type", "index", "delta", "logprobs"}, {"content"}),
    "content-end": ({"type", "index"}, None),
    "tool-plan-delta": ({"type", "delta"}, {"tool_plan"}),
    "tool-call-start": ({"type", "index", "delta"}, {"tool_calls"}),
    "tool-call-delta": ({"type", "index", "delta"}, {"tool_calls"}),
    "tool-call-end": ({"type", "index"}, None),
    "message-end": ({"type", "id", "delta"}, None),
}
MESSAGE_END_DELTA_FIELDS = {"error", "finish_reason", "usage"}

# --- /v2/embed ---
EMBED_REQUEST_FIELDS = {
    "model", "input_type", "texts", "images", "inputs", "max_tokens",
    "output_dimension", "embedding_types", "truncate", "priority",
}  # fmt: skip
INPUT_TYPES = {"search_document", "search_query", "classification", "clustering", "image"}
EMBED_RESPONSE_FIELDS = {"id", "embeddings", "texts", "images", "meta"}
# embed-v4.0's output dimensions; 1536 is the default.
EMBED_V4_DIMENSIONS = {256, 512, 1024, 1536}

CHAT_FIXTURES = sorted((FIXTURES_DIR / "cohere").glob("*.json"))
EMBED_FIXTURES = sorted((FIXTURES_DIR / "cohere_embed").glob("*.json"))


def _only(fields: dict, allowed: set, where: str) -> None:
    extra = set(fields) - allowed
    assert not extra, f"{where}: {sorted(extra)} not in the v2 reference"


def _content(role: str, content, where: str) -> None:
    if isinstance(content, str):
        return
    assert isinstance(content, list), f"{where}: content must be a string or a list"
    for block in content:
        allowed = CONTENT_BLOCKS[role]
        assert block.get("type") in allowed, f"{where}: {block.get('type')!r} block"
        _only(block, allowed[block["type"]], f"{where} {block['type']} block")


def _tool_call(call: dict, where: str) -> None:
    _only(call, {"id", "type", "function"}, where)
    assert call["type"] == "function" and call["id"], where
    _only(call["function"], {"name", "arguments"}, f"{where}.function")
    # ToolCallV2Function.arguments is a JSON string.
    assert isinstance(json.loads(call["function"]["arguments"]), dict), where


def check_chat_request(body: dict) -> None:
    _only(body, CHAT_REQUEST_FIELDS, "request")
    assert isinstance(body["stream"], bool) and body["model"]
    for i, msg in enumerate(body["messages"]):
        where = f"messages[{i}]"
        assert msg["role"] in MESSAGE_FIELDS, where
        _only(msg, MESSAGE_FIELDS[msg["role"]], where)
        if "content" in msg:
            _content(msg["role"], msg["content"], where)
        for call in msg.get("tool_calls", ()):
            _tool_call(call, f"{where}.tool_calls")
    for tool in body.get("tools", ()):
        _only(tool, {"type", "function"}, "tools[]")
        _only(tool["function"], {"name", "parameters", "description"}, "tools[].function")
        assert {"name", "parameters"} <= set(tool["function"])
    assert body.get("tool_choice", "REQUIRED") in TOOL_CHOICES
    assert len(body.get("stop_sequences", ())) <= MAX_STOP_SEQUENCES
    if "thinking" in body:
        _only(body["thinking"], {"type", "token_budget"}, "thinking")
        assert body["thinking"]["type"] in THINKING_TYPES
    if "response_format" in body:
        fmt = body["response_format"]
        assert fmt["type"] in ("text", "json_object")
        _only(fmt, {"type", "json_schema"} if fmt["type"] == "json_object" else {"type"}, "fmt")


def check_usage(usage: dict) -> None:
    _only(usage, USAGE_FIELDS, "usage")
    for key in ("billed_units", "tokens"):
        assert set(usage.get(key, {})) <= {"input_tokens", "output_tokens"}, usage


def check_chat_response(body: dict) -> None:
    _only(body, RESPONSE_FIELDS, "response")
    assert body["id"] and body["finish_reason"] in FINISH_REASONS
    msg = body["message"]
    _only(msg, RESPONSE_MESSAGE_FIELDS, "message")
    assert msg["role"] == "assistant"
    _content("assistant", msg.get("content", []), "message")
    for call in msg.get("tool_calls", ()):
        _tool_call(call, "message.tool_calls")
    assert (body["finish_reason"] == "TOOL_CALL") == bool(msg.get("tool_calls"))
    check_usage(body["usage"])


def check_stream(lines: list) -> None:
    """SSE: `event:` names its payload's type, each `data:` is one event,
    and blank lines end events."""
    events = []
    for name, data, blank in zip(lines[::3], lines[1::3], lines[2::3]):
        assert name.startswith("event: ") and data.startswith("data: ") and blank == ""
        event = json.loads(data[len("data: ") :])
        assert event["type"] == name[len("event: ") :]
        events.append(event)
    assert len(events) * 3 == len(lines)
    assert events[0]["type"] == "message-start" and events[0]["id"]
    assert events[-1]["type"] == "message-end"
    for event in events:
        fields, message_fields = STREAM_EVENTS[event["type"]]
        _only(event, fields, event["type"])
        if message_fields is not None:
            _only(event["delta"], {"message"}, f"{event['type']}.delta")
            _only(event["delta"]["message"], message_fields, f"{event['type']}.delta.message")
    end = events[-1]["delta"]
    _only(end, MESSAGE_END_DELTA_FIELDS, "message-end.delta")
    assert end["finish_reason"] in FINISH_REASONS
    check_usage(end["usage"])


@pytest.mark.parametrize("path", CHAT_FIXTURES, ids=lambda p: p.stem)
def test_chat_fixture_follows_the_reference(path):
    ex = Exchange.load(path)
    assert ex.url == "https://api.cohere.com/v2/chat"
    check_chat_request(ex.request_json)
    if ex.status != 200:
        _only(ex.response_json, ERROR_FIELDS, "error body")
        assert ex.response_json["message"]
    elif ex.lines is not None:
        assert ex.content_type == "text/event-stream" and ex.request_json["stream"] is True
        check_stream(ex.lines)
    else:
        assert ex.request_json["stream"] is False
        check_chat_response(ex.response_json)


@pytest.mark.parametrize("path", EMBED_FIXTURES, ids=lambda p: p.stem)
def test_embed_fixture_follows_the_reference(path):
    ex = Exchange.load(path)
    assert ex.url == "https://api.cohere.com/v2/embed"
    sent = ex.request_json
    _only(sent, EMBED_REQUEST_FIELDS, "embed request")
    assert sent["input_type"] in INPUT_TYPES and sent["embedding_types"] == ["float"]
    if ex.status != 200:
        _only(ex.response_json, ERROR_FIELDS, "error body")
        return
    body = ex.response_json
    _only(body, EMBED_RESPONSE_FIELDS, "embed response")
    vectors = body["embeddings"]["float"]
    assert len(vectors) == len(sent["texts"]) and body["texts"] == sent["texts"]
    assert {len(v) for v in vectors} <= EMBED_V4_DIMENSIONS
