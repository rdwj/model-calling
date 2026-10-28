"""Offline tests for OpenAI-compatible providers."""

import json

import pytest

from model_calling.core import (
    Base64Source,
    FileSource,
    Image,
    Message,
    Opaque,
    Request,
    ResponseFormat,
    Text,
    Thinking,
    ThinkingConfig,
    Tool,
    ToolCall,
    ToolChoice,
    ToolResult,
    UrlSource,
)
from model_calling.formats import TranslationError
from model_calling.errors import APIError
from model_calling.providers.openai_compat import (
    OpenAIProvider,
    OpenRouterProvider,
    VLLMProvider,
)
from tests.harness.replay import fixture_path


def load_fixture(backend: str, scenario: str) -> dict:
    return json.loads(fixture_path(backend, scenario).read_text())


def req(*messages, **kwargs):
    """Build a Request with default model and messages."""
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
    return VLLMProvider(model="gpt-oss-20b", base_url="http://localhost:8000/v1")


@pytest.fixture
def openai_provider():
    return OpenAIProvider(model="gpt-4", base_url="https://api.openai.com/v1")


@pytest.fixture
def openrouter_provider():
    return OpenRouterProvider(
        model="mistralai/mistral-nemo", base_url="https://openrouter.ai/api/v1"
    )


# --- endpoint ---


@pytest.mark.parametrize(
    "base_url,expected",
    [
        ("http://localhost:8000/v1", "http://localhost:8000/v1/chat/completions"),
        ("http://localhost:8000/v1/", "http://localhost:8000/v1/chat/completions"),
    ],
)
def test_endpoint(base_url, expected):
    provider = VLLMProvider(model="m", base_url=base_url)
    assert provider.endpoint() == expected


# --- build_request: system messages ---


def test_build_request_system_in_system_field(provider):
    body = provider.build_request(
        req(user(Text(text="Hi")), system=[Text(text="You are helpful.")])
    )
    assert body["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert body["messages"][1] == {"role": "user", "content": "Hi"}


def test_build_request_mid_conversation_system_message(provider):
    body = provider.build_request(
        req(user(Text(text="Hi")), system(Text(text="Be brief.")), user(Text(text="Tell me more")))
    )
    assert body["messages"][1] == {"role": "system", "content": "Be brief."}


# --- build_request: content formats ---


def test_build_request_user_text_as_string(provider):
    body = provider.build_request(req(user(Text(text="Hello"))))
    assert body["messages"][0]["content"] == "Hello"


def test_build_request_user_multiple_text_blocks_as_parts(provider):
    body = provider.build_request(req(user(Text(text="Hello"), Text(text=" world"))))
    assert body["messages"][0]["content"] == [
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": " world"},
    ]


@pytest.mark.parametrize(
    "source,expected_url,extras,expected_image_url",
    [
        (
            Base64Source(media_type="image/png", data="iVBORw0KG"),
            "data:image/png;base64,iVBORw0KG",
            {},
            {"url": "data:image/png;base64,iVBORw0KG"},
        ),
        (
            UrlSource(url="https://example.com/image.png"),
            "https://example.com/image.png",
            {},
            {"url": "https://example.com/image.png"},
        ),
        (
            UrlSource(url="https://example.com/image.png"),
            "https://example.com/image.png",
            {"openai_chat": {"detail": "low"}},
            {"url": "https://example.com/image.png", "detail": "low"},
        ),
    ],
)
def test_build_request_images(provider, source, expected_url, extras, expected_image_url):
    img = Image(source=source, extras=extras)
    body = provider.build_request(req(user(Text(text="What is this?"), img)))
    content = body["messages"][0]["content"]
    assert content[1] == {"type": "image_url", "image_url": expected_image_url}


def test_build_request_file_source_image_raises(provider):
    img = Image(source=FileSource(file_id="file-123"))
    with pytest.raises(TranslationError, match="cannot accept an image by file id"):
        provider.build_request(req(user(img)))


# --- build_request: tool results ---


def test_build_request_tool_result_as_tool_message(provider):
    result = ToolResult(tool_call_id="call_123", content=[Text(text="Success")])
    body = provider.build_request(
        req(assistant(ToolCall(id="call_123", name="tool", input={})), user(result))
    )
    assert body["messages"][1] == {"role": "tool", "tool_call_id": "call_123", "content": "Success"}


def test_build_request_tool_result_non_text_raises(provider):
    result = ToolResult(
        tool_call_id="call_123",
        content=[Image(source=UrlSource(url="https://example.com/result.png"))],
    )
    with pytest.raises(TranslationError, match="tool results may contain only text"):
        provider.build_request(req(user(result)))


# --- build_request: assistant messages ---


@pytest.mark.parametrize(
    "provider_cls,field",
    [(VLLMProvider, "reasoning"), (OpenRouterProvider, "reasoning"), (OpenAIProvider, None)],
)
def test_build_request_assistant_thinking_field(provider_cls, field):
    p = provider_cls(model="m", base_url="http://localhost/v1")
    body = p.build_request(
        req(assistant(Thinking(thinking="Let me think..."), Text(text="Answer")))
    )
    msg = body["messages"][0]
    if field:
        assert msg[field] == "Let me think..."
    else:
        assert "reasoning" not in msg
    assert msg["content"] == "Answer"


def test_build_request_assistant_tool_calls(provider):
    call = ToolCall(id="call_1", name="get_weather", input={"location": "Paris"})
    body = provider.build_request(req(assistant(call)))
    assert body["messages"][0]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
        }
    ]
    assert body["messages"][0]["content"] is None


def test_build_request_assistant_tool_call_raw_arguments(provider):
    call = ToolCall(id="call_1", name="broken_tool", input={}, raw_arguments="invalid json {")
    body = provider.build_request(req(assistant(call)))
    assert body["messages"][0]["tool_calls"][0]["function"]["arguments"] == "invalid json {"


def test_build_request_assistant_text_and_tool_calls(provider):
    body = provider.build_request(
        req(
            assistant(
                Text(text="I'll check the weather."),
                ToolCall(id="call_1", name="get_weather", input={"location": "Paris"}),
            )
        )
    )
    msg = body["messages"][0]
    assert msg["content"] == "I'll check the weather."
    assert len(msg["tool_calls"]) == 1


def test_build_request_assistant_empty_string_when_no_content_or_tool_calls(provider):
    body = provider.build_request(req(assistant()))
    assert body["messages"][0]["content"] == ""


# --- build_request: tools ---


def test_build_request_tools(provider):
    body = provider.build_request(
        req(
            tools=[
                Tool(
                    name="get_weather",
                    description="Get weather",
                    input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
                )
            ]
        )
    )
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]


def test_build_request_tools_with_extras(provider):
    body = provider.build_request(
        req(
            tools=[
                Tool(
                    name="get_weather",
                    input_schema={"type": "object"},
                    extras={"openai_chat": {"strict": True}},
                )
            ]
        )
    )
    assert body["tools"][0]["function"]["strict"] is True


# --- build_request: tool_choice ---


@pytest.mark.parametrize(
    "mode,expected",
    [("auto", "auto"), ("none", "none"), ("any", "required")],
)
def test_build_request_tool_choice_modes(provider, mode, expected):
    tools = [Tool(name="get_weather", input_schema={"type": "object"})] if mode == "auto" else None
    body = provider.build_request(req(tools=tools, tool_choice=ToolChoice(mode=mode)))
    assert body["tool_choice"] == expected


def test_build_request_tool_choice_tool(provider):
    body = provider.build_request(req(tool_choice=ToolChoice(mode="tool", name="get_weather")))
    assert body["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_build_request_parallel_tool_calls(provider):
    body = provider.build_request(req(tool_choice=ToolChoice(mode="auto", parallel=False)))
    assert body["parallel_tool_calls"] is False


# --- build_request: response_format ---


def test_build_request_response_format_json_object(provider):
    body = provider.build_request(req(response_format=ResponseFormat(type="json_object")))
    assert body["response_format"] == {"type": "json_object"}


def test_build_request_response_format_json_schema(provider):
    body = provider.build_request(
        req(
            response_format=ResponseFormat(
                type="json_schema",
                name="person",
                json_schema={"type": "object", "properties": {"name": {"type": "string"}}},
                strict=True,
            )
        )
    )
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
            "strict": True,
        },
    }


def test_build_request_response_format_json_schema_default_name(provider):
    body = provider.build_request(
        req(response_format=ResponseFormat(type="json_schema", json_schema={"type": "object"}))
    )
    assert body["response_format"]["json_schema"]["name"] == "response"


# --- build_request: thinking config ---


@pytest.mark.parametrize(
    "thinking,expected",
    [
        (ThinkingConfig(mode="disabled"), "none"),
        (ThinkingConfig(mode="enabled", effort="high"), "high"),
        (ThinkingConfig(mode="enabled"), None),
    ],
)
def test_build_request_thinking_effort(provider, thinking, expected):
    body = provider.build_request(req(thinking=thinking))
    if expected is None:
        assert "reasoning_effort" not in body
    else:
        assert body["reasoning_effort"] == expected


# --- build_request: top_k ---


def test_build_request_top_k_vllm(provider):
    body = provider.build_request(req(top_k=40))
    assert body["top_k"] == 40


def test_build_request_top_k_openai_raises(openai_provider):
    with pytest.raises(TranslationError, match="does not support top_k"):
        openai_provider.build_request(req(top_k=40))


# --- build_request: max_tokens field ---


@pytest.mark.parametrize(
    "provider_cls,field",
    [(VLLMProvider, "max_tokens"), (OpenAIProvider, "max_completion_tokens")],
)
def test_build_request_max_tokens_field(provider_cls, field):
    p = provider_cls(model="m", base_url="http://localhost/v1")
    body = p.build_request(req(max_tokens=100))
    assert body[field] == 100
    other_field = "max_completion_tokens" if field == "max_tokens" else "max_tokens"
    assert other_field not in body


# --- build_request: sampling params ---


@pytest.mark.parametrize(
    "param,value",
    [
        ("temperature", 0.7),
        ("top_p", 0.9),
        ("seed", 42),
        ("presence_penalty", 0.5),
        ("frequency_penalty", 0.3),
        ("stop", ["STOP", "END"]),
    ],
)
def test_build_request_sampling_params(provider, param, value):
    body = provider.build_request(req(**{param: value}))
    assert body[param] == value


# --- build_request: stream ---


def test_build_request_stream_adds_stream_options(provider):
    body = provider.build_request(req(stream=True))
    assert body["stream_options"] == {"include_usage": True}


# --- build_request: extras passthrough ---


def test_build_request_extras_passthrough(provider):
    body = provider.build_request(
        req(extras={"openai_chat": {"user": "user-123", "logprobs": True}})
    )
    assert body["user"] == "user-123"
    assert body["logprobs"] is True


def test_build_request_extras_stream_options_not_overridden(provider):
    body = provider.build_request(
        req(stream=True, extras={"openai_chat": {"stream_options": {"include_usage": False}}})
    )
    assert body["stream_options"] == {"include_usage": True}


# --- build_request: opaque blocks ---


def test_build_request_opaque_block_foreign_format_raises(provider):
    opaque = Opaque(format="anthropic", data={"type": "text", "text": "Hi"})
    with pytest.raises(TranslationError, match="cannot accept a anthropic"):
        provider.build_request(req(user(opaque)))


# --- parse_response: fixtures ---


def test_parse_response_vllm_text(provider):
    fixture = load_fixture("vllm", "text")
    body = fixture["response"]["json"]
    expected_reasoning = body["choices"][0]["message"]["reasoning"]
    resp = provider.parse_response(body)

    assert len(resp.content) == 2
    assert isinstance(resp.content[0], Thinking)
    assert resp.content[0].thinking == expected_reasoning
    assert isinstance(resp.content[1], Text)
    assert resp.content[1].text == "pong"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == body["usage"]["prompt_tokens"]
    assert resp.usage.output_tokens == body["usage"]["completion_tokens"]


def test_parse_response_vllm_tools(provider):
    fixture = load_fixture("vllm", "tools")
    body = fixture["response"]["json"]
    expected_reasoning = body["choices"][0]["message"]["reasoning"]
    expected_call = body["choices"][0]["message"]["tool_calls"][0]
    resp = provider.parse_response(body)

    thinking = next(b for b in resp.content if isinstance(b, Thinking))
    assert thinking.thinking == expected_reasoning

    call = next(b for b in resp.content if isinstance(b, ToolCall))
    assert call.id == expected_call["id"]
    assert call.name == "get_weather"
    assert call.input == {"location": "Paris"}
    assert resp.stop_reason == "tool_use"


def test_parse_response_openrouter_text(openrouter_provider):
    fixture = load_fixture("openai_compat", "text")
    body = fixture["response"]["json"]
    resp = openrouter_provider.parse_response(body)

    assert isinstance(resp.content[0], Text)
    expected_text = body["choices"][0]["message"]["content"]
    assert resp.content[0].text == expected_text
    assert resp.content[0].text.lower() == "pong"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == body["usage"]["prompt_tokens"]
    assert resp.usage.output_tokens == body["usage"]["completion_tokens"]


def test_parse_response_openrouter_tools(openrouter_provider):
    fixture = load_fixture("openai_compat", "tools")
    body = fixture["response"]["json"]
    expected_call = body["choices"][0]["message"]["tool_calls"][0]
    resp = openrouter_provider.parse_response(body)

    call = next(b for b in resp.content if isinstance(b, ToolCall))
    assert call.id == expected_call["id"]
    assert call.name == "get_weather"
    assert call.input == {"location": "Paris"}
    assert resp.stop_reason == "tool_use"


# --- parse_response: synthetic ---


def test_parse_response_reasoning_content_accepted(provider):
    body = {
        "id": "test",
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Answer",
                    "reasoning_content": "Old vLLM field",
                },
                "finish_reason": "stop",
            }
        ],
    }
    resp = provider.parse_response(body)
    assert isinstance(resp.content[0], Thinking)
    assert resp.content[0].thinking == "Old vLLM field"


def test_parse_response_refusal(provider):
    body = {
        "id": "test",
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "refusal": "I cannot help with that.",
                },
                "finish_reason": "stop",
            }
        ],
    }
    resp = provider.parse_response(body)
    assert isinstance(resp.content[0], Text)
    assert resp.content[0].text == "I cannot help with that."
    assert resp.stop_reason == "refusal"


@pytest.mark.parametrize(
    "finish_reason,has_tool_calls,expected_stop_reason",
    [
        ("stop", False, "end_turn"),
        ("stop", True, "tool_use"),
        ("length", False, "max_tokens"),
        ("tool_calls", False, "tool_use"),
        ("content_filter", False, "refusal"),
    ],
)
def test_parse_response_finish_reason_mapping(
    provider, finish_reason, has_tool_calls, expected_stop_reason
):
    body = {
        "id": "test",
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Text"},
                "finish_reason": finish_reason,
            }
        ],
    }
    if has_tool_calls:
        body["choices"][0]["message"]["content"] = None
        body["choices"][0]["message"]["tool_calls"] = [
            {"id": "call_1", "type": "function", "function": {"name": "tool", "arguments": "{}"}}
        ]
    resp = provider.parse_response(body)
    assert resp.stop_reason == expected_stop_reason


def test_parse_response_error_without_choices_raises(provider):
    body = {"error": {"message": "Model not found", "type": "invalid_request_error"}, "choices": []}
    with pytest.raises(APIError, match="returned an error"):
        provider.parse_response(body)


def test_parse_response_empty_choices_raises(provider):
    body = {"id": "test", "model": "m", "choices": []}
    with pytest.raises(TranslationError, match="no choices"):
        provider.parse_response(body)


def test_parse_response_usage_details(provider):
    body = {
        "id": "test",
        "model": "m",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
    }
    resp = provider.parse_response(body)
    assert resp.usage.input_tokens == 10
    assert resp.usage.output_tokens == 5
    assert resp.usage.cache_read_tokens == 3
    assert resp.usage.cache_write_tokens == 2
    assert resp.usage.reasoning_tokens == 1


def test_build_request_keeps_system_blocks_apart(provider):
    body = provider.build_request(req(system=[Text(text="Be kind."), Text(text="Be brief.")]))
    assert body["messages"][0] == {"role": "system", "content": "Be kind.\nBe brief."}


def test_build_request_top_k_in_extras_is_still_refused_by_openai(openai_provider):
    with pytest.raises(TranslationError, match="does not support top_k"):
        openai_provider.build_request(req(extras={"openai_chat": {"top_k": 5}}))


@pytest.mark.parametrize(
    "body",
    [
        {"model": None, "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]},
        {"choices": [{"message": None, "finish_reason": "stop"}]},
    ],
    ids=["null_model", "null_message"],
)
def test_parse_response_tolerates_nulls(provider, body):
    resp = provider.parse_response(body)
    assert resp.model == "gpt-oss-20b"
    assert resp.stop_reason == "end_turn"
