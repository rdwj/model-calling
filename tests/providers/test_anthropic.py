"""Offline tests for the Anthropic Messages provider."""

import json

import pytest
from anthropic.types import Message as SDKMessage

from model_calling.core import (
    CacheControl,
    Message,
    Opaque,
    RedactedThinking,
    Request,
    ResponseFormat,
    Text,
    Thinking,
    ThinkingConfig,
    ToolCall,
)
from model_calling.formats import TranslationError, anthropic, chat_completions
from model_calling.providers.anthropic import (
    API_VERSION,
    DEFAULT_MAX_TOKENS,
    AnthropicProvider,
)
from tests.harness.replay import fixture_path


def load_fixture(backend: str, scenario: str) -> dict:
    return json.loads(fixture_path(backend, scenario).read_text())


def req(*messages, **kwargs):
    if not messages:
        messages = (user(Text(text="Hi")),)
    return Request(model=kwargs.pop("model", "m"), messages=list(messages), **kwargs)


def user(*blocks):
    return Message(role="user", content=list(blocks))


def assistant(*blocks):
    return Message(role="assistant", content=list(blocks))


def system(*blocks):
    return Message(role="system", content=list(blocks))


@pytest.fixture
def provider():
    return AnthropicProvider(model="claude-sonnet-4", base_url="https://api.anthropic.com/v1")


# --- endpoint and headers ---


@pytest.mark.parametrize(
    "base_url,expected",
    [
        ("https://api.anthropic.com/v1", "https://api.anthropic.com/v1/messages"),
        ("https://api.anthropic.com/v1/", "https://api.anthropic.com/v1/messages"),
    ],
)
def test_endpoint(base_url, expected):
    p = AnthropicProvider(model="m", base_url=base_url)
    assert p.endpoint() == expected


@pytest.mark.parametrize(
    "api_key,extra,expected_keys",
    [
        ("sk-ant-123", {}, {"authorization", "anthropic-version"}),
        (None, {}, {"anthropic-version"}),
        (
            "sk-ant-123",
            {"X-Custom": "val"},
            {"authorization", "anthropic-version", "X-Custom"},
        ),
    ],
)
def test_headers(api_key, extra, expected_keys):
    p = AnthropicProvider(
        model="m",
        base_url="https://api.anthropic.com/v1",
        api_key=api_key,
        extra_headers=extra,
    )
    headers = p.headers()
    assert set(headers.keys()) == expected_keys
    assert headers["anthropic-version"] == API_VERSION
    if api_key:
        assert headers["authorization"] == f"Bearer {api_key}"


# --- build_request ---


def test_build_request_replaces_model(provider):
    body = provider.build_request(req(model="other"))
    assert body["model"] == "claude-sonnet-4"


def test_build_request_messages_body_passthrough(provider):
    internal_req = anthropic.request_to_internal(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "Hi"}],
            "system": "Be helpful.",
            "max_tokens": 1024,
            "temperature": 0.7,
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
    )
    body = provider.build_request(internal_req)
    assert body["model"] == "claude-sonnet-4"
    assert body["messages"] == [{"role": "user", "content": "Hi"}]
    assert body["system"] == "Be helpful."
    assert body["max_tokens"] == 1024
    assert body["thinking"]["budget_tokens"] == 2000


def test_build_request_shorthand_preserved(provider):
    internal_req = anthropic.request_to_internal(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }
    )
    body = provider.build_request(internal_req)
    assert body["messages"][0] == {"role": "user", "content": "Hello"}


def test_build_request_system_with_cache_control(provider):
    sys = [Text(text="Helpful.", cache_control=CacheControl(ttl="5m"))]
    body = provider.build_request(req(user(Text(text="Hi")), system=sys))
    assert body["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}


@pytest.mark.parametrize(
    "content,expected_content_len",
    [
        ([Thinking(thinking="Unsigned", signature=""), Text(text="Hi")], 1),
        ([Thinking(thinking="Unsigned", signature=None), Text(text="Hi")], 1),
        ([Thinking(thinking="Signed", signature="sig"), Text(text="Hi")], 2),
        ([RedactedThinking(data="enc"), Text(text="Hi")], 2),
    ],
)
def test_build_request_thinking_in_assistant_turns(provider, content, expected_content_len):
    internal_req = req(assistant(*content))
    body = provider.build_request(internal_req)
    assert len(body["messages"][0]["content"]) == expected_content_len


def test_build_request_drops_a_turn_left_empty(provider):
    """An assistant turn that was only unsigned thinking (an Ollama model
    that ran out of tokens while thinking) goes, rather than going empty."""
    hi = user(Text(text="Hi"))
    body = provider.build_request(req(hi, assistant(Thinking(thinking="hmm")), hi))
    assert [m["role"] for m in body["messages"]] == ["user", "user"]


def test_build_request_omits_tool_choice_without_tools(provider):
    chat_req = chat_completions.request_to_internal(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "Hi"}],
            "parallel_tool_calls": False,
        }
    )
    assert "tool_choice" not in provider.build_request(chat_req)


def test_build_request_max_tokens_defaults(provider):
    internal_req = req(max_tokens=None)
    body = provider.build_request(internal_req)
    assert body["max_tokens"] == DEFAULT_MAX_TOKENS


def test_build_request_max_tokens_from_chat_completions(provider):
    chat_req = chat_completions.request_to_internal(
        {"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}
    )
    assert chat_req.max_tokens is None
    body = provider.build_request(chat_req)
    assert body["max_tokens"] == DEFAULT_MAX_TOKENS


@pytest.mark.parametrize(
    "effort,expected_thinking,expected_effort",
    [
        ("low", {"type": "adaptive"}, "low"),
        ("medium", {"type": "adaptive"}, "medium"),
        ("high", {"type": "adaptive"}, "high"),
        ("none", {"type": "disabled"}, None),
    ],
)
def test_build_request_reasoning_effort(provider, effort, expected_thinking, expected_effort):
    chat_req = chat_completions.request_to_internal(
        {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "reasoning_effort": effort,
        }
    )
    body = provider.build_request(chat_req)
    assert body["thinking"] == expected_thinking
    if expected_effort:
        assert body["output_config"]["effort"] == expected_effort


def test_build_request_reasoning_effort_unsupported(provider):
    internal_req = req(
        thinking=ThinkingConfig(mode="enabled", effort="minimal", budget_tokens=None)
    )
    assert "thinking.effort" in provider.unsupported_params(internal_req)
    body = provider.build_request(internal_req)
    # Asked for minimal reasoning: no thinking, not the model's default.
    assert "thinking" not in body and "output_config" not in body


@pytest.mark.parametrize(
    "param,value,in_unsupported",
    [
        ("seed", 42, True),
        ("presence_penalty", 0.5, True),
        ("frequency_penalty", 0.3, True),
        ("response_format", ResponseFormat(type="json_object"), True),
        (
            "response_format",
            ResponseFormat(type="json_schema", json_schema={"type": "object"}),
            False,
        ),
    ],
)
def test_unsupported_params(provider, param, value, in_unsupported):
    internal_req = req(**{param: value})
    unsupported = provider.unsupported_params(internal_req)
    assert (param in unsupported) == in_unsupported


def test_unsupported_chat_completions_extras(provider):
    """Chat Completions extras that Anthropic doesn't support: logprobs,
    logit_bias, and user (Phase 12 item 1)."""
    chat_req = chat_completions.request_to_internal(
        {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "logprobs": True,
            "logit_bias": {"123": 10},
            "user": "user123",
        }
    )
    unsupported = provider.unsupported_params(chat_req)
    assert "logprobs" in unsupported
    assert "logit_bias" in unsupported
    assert "user" in unsupported


def test_build_request_opaque_tool_raises(provider):
    opaque = Opaque(format="openai_chat", data={"type": "code_interpreter"})
    with pytest.raises(TranslationError, match="cannot send a 'openai_chat' block"):
        provider.build_request(req(tools=[opaque]))


# --- parse_response ---


def test_parse_response_text_fixture(provider):
    fixture = load_fixture("vllm_messages", "text")
    body = fixture["response"]["json"]
    resp = provider.parse_response(body)
    assert resp.id == body["id"]
    assert len(resp.content) == 2
    assert isinstance(resp.content[0], Thinking)
    assert isinstance(resp.content[1], Text)
    assert resp.content[1].text == "pong"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens > 0


def test_parse_response_tools_fixture(provider):
    fixture = load_fixture("vllm_messages", "tools")
    body = fixture["response"]["json"]
    resp = provider.parse_response(body)
    call = next(b for b in resp.content if isinstance(b, ToolCall))
    assert call.name == "get_weather"
    assert call.input == {"location": "Paris"}
    assert resp.stop_reason == "tool_use"


def test_parse_response_stop_details_and_container_reach_messages_egress(provider):
    stop_details = {"type": "refusal", "category": "cyber", "explanation": "Declined."}
    container = {"id": "container_1", "expires_at": "2026-10-30T00:00:00Z", "skills": None}
    body = {
        **load_fixture("vllm_messages", "text")["response"]["json"],
        "stop_details": stop_details,
        "container": container,
    }
    out = anthropic.response_from_internal(provider.parse_response(body))
    assert (out["stop_details"], out["container"]) == (stop_details, container)
    sdk = SDKMessage.model_validate(out)
    assert (sdk.stop_details.category, sdk.container.id) == ("cyber", "container_1")


def test_parse_response_usage_with_cache(provider):
    body = {
        "id": "msg",
        "type": "message",
        "role": "assistant",
        "model": "m",
        "content": [{"type": "text", "text": "Hi"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 2,
        },
    }
    resp = provider.parse_response(body)
    assert resp.usage.input_tokens == 15
    assert resp.usage.cache_read_tokens == 3
    assert resp.usage.cache_write_tokens == 2


@pytest.mark.parametrize(
    "wire_reason,expected",
    [
        ("end_turn", "end_turn"),
        ("max_tokens", "max_tokens"),
        ("tool_use", "tool_use"),
        ("model_context_window_exceeded", "context_window_exceeded"),
        ("future_reason", None),
    ],
)
def test_parse_response_stop_reason_mapping(provider, wire_reason, expected):
    body = {
        "id": "msg",
        "type": "message",
        "role": "assistant",
        "model": "m",
        "content": [{"type": "text", "text": "Hi"}],
        "stop_reason": wire_reason,
    }
    resp = provider.parse_response(body)
    assert resp.stop_reason == expected


@pytest.mark.parametrize(
    "stop_reason,has_tool,expected",
    [
        ("end_turn", False, "end_turn"),
        ("end_turn", True, "tool_use"),
        ("max_tokens", True, "max_tokens"),
    ],
)
def test_parse_response_tool_use_override(provider, stop_reason, has_tool, expected):
    content = [{"type": "text", "text": "Hi"}]
    if has_tool:
        content.append({"type": "tool_use", "id": "call", "name": "tool", "input": {}})
    body = {
        "id": "msg",
        "type": "message",
        "role": "assistant",
        "model": "m",
        "content": content,
        "stop_reason": stop_reason,
    }
    resp = provider.parse_response(body)
    assert resp.stop_reason == expected


def test_parse_response_tool_named_fixture(provider):
    fixture = load_fixture("vllm_messages", "tool_named")
    body = fixture["response"]["json"]
    assert body["stop_reason"] == "end_turn"
    resp = provider.parse_response(body)
    assert resp.stop_reason == "tool_use"
