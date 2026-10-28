"""
Anthropic Messages ↔ internal format: the Phase 2 design checkpoint.

Round trips must be lossless, and both ends must validate against the
anthropic SDK types.
"""

import copy

import pytest

from model_calling.core import (
    Image,
    Opaque,
    RedactedThinking,
    Text,
    Thinking,
    Tool,
    ToolCall,
    ToolResult,
)
from model_calling.formats import TranslationError
from model_calling.formats import anthropic as fmt
from tests.harness.sdk import validate_anthropic_message, validate_anthropic_request

EPHEMERAL = {"type": "ephemeral"}
EPHEMERAL_1H = {"type": "ephemeral", "ttl": "1h"}
WEATHER_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
PNG_1PX = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

REQUEST = {
    "model": "claude-sonnet-5-5",
    "max_tokens": 1024,
    "system": [{"type": "text", "text": "You are terse.", "cache_control": EPHEMERAL_1H}],
    "metadata": {"user_id": "u-123"},
    "thinking": {"type": "enabled", "budget_tokens": 2048},
    "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
    "tools": [
        {
            "name": "get_weather",
            "description": "Current weather for a city",
            "input_schema": WEATHER_SCHEMA,
            "cache_control": EPHEMERAL,
        },
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 2},
    ],
    "messages": [
        {"role": "user", "content": "Weather in Paris and Oslo?"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Two cities, two calls.", "signature": "sig-abc"},
                {"type": "redacted_thinking", "data": "opaque-redacted"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_2",
                    "name": "get_weather",
                    "input": {"city": "Oslo"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "18C, clear"},
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_2",
                    "is_error": True,
                    "content": [{"type": "text", "text": "upstream timeout"}],
                    "cache_control": EPHEMERAL,
                },
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": PNG_1PX},
                },
                {
                    "type": "document",
                    "source": {"type": "text", "media_type": "text/plain", "data": "Oslo notes"},
                    "citations": {"enabled": True},
                },
                {"type": "text", "text": "Retry Oslo.", "cache_control": EPHEMERAL},
            ],
        },
    ],
}

RESPONSE = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "content": [
        {"type": "thinking", "thinking": "Retry the failed call.", "signature": "sig-def"},
        {"type": "redacted_thinking", "data": "opaque-redacted-2"},
        {
            "type": "text",
            "text": "Paris is 18C.",
            "citations": [
                {
                    "type": "char_location",
                    "cited_text": "18C",
                    "document_index": 0,
                    "document_title": None,
                    "start_char_index": 0,
                    "end_char_index": 3,
                }
            ],
        },
        {
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": {"query": "Oslo"},
        },
        {"type": "tool_use", "id": "toolu_3", "name": "get_weather", "input": {"city": "Oslo"}},
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 120,
        "output_tokens": 40,
        "cache_read_input_tokens": 100,
        "cache_creation_input_tokens": 20,
        "service_tier": "standard",
    },
}


def test_request_round_trip_is_lossless():
    original = validate_anthropic_request(copy.deepcopy(REQUEST))
    out = fmt.request_from_internal(fmt.request_to_internal(original))
    assert validate_anthropic_request(out) == original


def test_request_decodes_to_typed_blocks():
    req = fmt.request_to_internal(REQUEST)
    assistant, results = req.messages[1].content, req.messages[2].content
    assert [type(b) for b in assistant] == [Thinking, RedactedThinking, ToolCall, ToolCall]
    assert assistant[0].signature == "sig-abc"
    assert [type(b) for b in results] == [ToolResult, ToolResult, Image, Opaque, Text]
    assert (results[0].content_shorthand, results[0].is_error) == (True, None)
    assert (results[1].is_error, results[1].cache_control.ttl) == (True, None)
    assert results[4].cache_control is not None
    assert req.system[0].cache_control.ttl == "1h"
    assert [type(t) for t in req.tools] == [Tool, Opaque]
    assert req.tool_choice.parallel is False
    assert (req.thinking.mode, req.thinking.budget_tokens) == ("enabled", 2048)
    assert req.extras == {"anthropic": {"metadata": {"user_id": "u-123"}}}


def test_response_round_trip_is_lossless():
    original = validate_anthropic_message(RESPONSE)
    out = fmt.response_from_internal(fmt.response_to_internal(RESPONSE))
    assert out == RESPONSE
    assert validate_anthropic_message(out) == original


def test_response_decodes_to_typed_blocks():
    resp = fmt.response_to_internal(RESPONSE)
    assert [type(b) for b in resp.content] == [Thinking, RedactedThinking, Text, Opaque, ToolCall]
    assert "citations" in resp.content[2].extras["anthropic"]
    assert resp.stop_reason == "tool_use"
    # Internal input_tokens includes the cached tokens Anthropic counts separately.
    assert (resp.usage.input_tokens, resp.usage.cache_read_tokens) == (240, 100)


@pytest.mark.parametrize(
    "wire, internal",
    [
        ("model_context_window_exceeded", "context_window_exceeded"),
        ("end_turn", "end_turn"),
        # Newer than this module: kept in extras and restored on the way out.
        ("future_reason", None),
    ],
)
def test_stop_reason_round_trip(wire, internal):
    body = {**RESPONSE, "stop_reason": wire}
    resp = fmt.response_to_internal(body)
    assert resp.stop_reason == internal
    assert fmt.response_from_internal(resp) == body


def test_foreign_opaque_block_is_rejected():
    resp = fmt.response_to_internal(RESPONSE)
    resp.content.append(Opaque(format="openai_chat", data={"type": "refusal"}))
    with pytest.raises(TranslationError, match="'openai_chat' block of type 'refusal'"):
        fmt.response_from_internal(resp)


def test_system_rejects_non_text_blocks():
    body = {**REQUEST, "system": [{"type": "image", "source": {"type": "url", "url": "x"}}]}
    with pytest.raises(TranslationError, match="only text blocks"):
        fmt.request_to_internal(body)


def user_turn(*blocks) -> dict:
    return {**REQUEST, "messages": [{"role": "user", "content": list(blocks)}]}


# Each body must survive the round trip unchanged. Explicit nulls and values
# newer than the translator are kept verbatim in extras.
EDGE_REQUESTS = {
    "null_cache_control": user_turn({"type": "text", "text": "x", "cache_control": None}),
    "unknown_cache_ttl": user_turn(
        {"type": "text", "text": "x", "cache_control": {"type": "ephemeral", "ttl": "24h"}}
    ),
    "unknown_cache_type": user_turn(
        {"type": "text", "text": "x", "cache_control": {"type": "persistent"}}
    ),
    "null_is_error": user_turn({"type": "tool_result", "tool_use_id": "t", "is_error": None}),
    "no_tool_result_content": user_turn({"type": "tool_result", "tool_use_id": "t"}),
    "empty_tool_result_content": user_turn(
        {"type": "tool_result", "tool_use_id": "t", "content": []}
    ),
    "opaque_in_tool_result": user_turn(
        {
            "type": "tool_result",
            "tool_use_id": "t",
            "content": [{"type": "search_result", "source": "s", "title": "t", "content": []}],
        }
    ),
    "url_image": user_turn({"type": "image", "source": {"type": "url", "url": "https://x/a.png"}}),
    "file_image": user_turn({"type": "image", "source": {"type": "file", "file_id": "file_1"}}),
    "null_temperature": {**REQUEST, "temperature": None},
    "null_thinking": {**REQUEST, "thinking": None},
    "unknown_thinking": {**REQUEST, "thinking": {"type": "between_tools"}},
    "unknown_tool_choice": {**REQUEST, "tool_choice": {"type": "auto_v2"}},
    "system_shorthand": {**REQUEST, "system": "Be terse."},
    "streaming": {**REQUEST, "stream": True},
}


@pytest.mark.parametrize("body", EDGE_REQUESTS.values(), ids=EDGE_REQUESTS.keys())
def test_request_edge_cases_round_trip(body):
    out = fmt.request_from_internal(fmt.request_to_internal(copy.deepcopy(body)))
    assert out == body


EDGE_RESPONSES = {
    "null_cache_usage": {
        **RESPONSE,
        "usage": {
            "input_tokens": 5,
            "output_tokens": 1,
            "cache_read_input_tokens": None,
            "cache_creation_input_tokens": None,
        },
    },
    "null_stop_reason": {**RESPONSE, "stop_reason": None},
    "text_with_null_citations": {
        **RESPONSE,
        "content": [{"type": "text", "text": "hi", "citations": None}],
    },
}


@pytest.mark.parametrize("body", EDGE_RESPONSES.values(), ids=EDGE_RESPONSES.keys())
def test_response_edge_cases_round_trip(body):
    validate_anthropic_message(body)
    assert fmt.response_from_internal(fmt.response_to_internal(body)) == body


@pytest.mark.parametrize("wire", [None, "future_reason"])
def test_typed_stop_reason_overrides_extras(wire):
    resp = fmt.response_to_internal({**RESPONSE, "stop_reason": wire})
    resp.stop_reason = "end_turn"
    assert fmt.response_from_internal(resp)["stop_reason"] == "end_turn"


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": [{"type": "text"}]},
        {"role": "user", "content": ["bare string block"]},
        {"role": "user", "content": [{"type": "image", "source": {"type": "base64"}}]},
        {"role": "user", "content": [{"type": "tool_use", "id": "t", "name": "n", "input": 1}]},
    ],
)
def test_malformed_request_raises_translation_error(message):
    with pytest.raises(TranslationError, match="invalid Anthropic request"):
        fmt.request_to_internal({**REQUEST, "messages": [message]})
