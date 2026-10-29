"""Offline tests for OpenAI Chat Completions ingress translation."""

import json

import pytest

from model_calling.core import (
    Base64Source,
    Image,
    Response,
    Text,
    Thinking,
    ToolCall,
    ToolResult,
    Usage,
    UrlSource,
)
from model_calling.formats import TranslationError
from model_calling.formats import chat_completions as chat
from tests.harness.sdk import validate_chat_completion

# --- request_to_internal tests ---


@pytest.mark.parametrize(
    "messages,expected_system,expected_first_role",
    [
        # Leading system and developer messages become Request.system
        (
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
            [Text(text="You are helpful.")],
            "user",
        ),
        (
            [
                {"role": "developer", "content": "System rules"},
                {"role": "user", "content": "Hi"},
            ],
            [Text(text="System rules")],
            "user",
        ),
        # Multiple leading system/developer messages concatenate
        (
            [
                {"role": "system", "content": "Rule 1"},
                {"role": "developer", "content": "Rule 2"},
                {"role": "user", "content": "Hi"},
            ],
            [Text(text="Rule 1"), Text(text="Rule 2")],
            "user",
        ),
        # Mid-conversation system stays a message
        (
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "system", "content": "New rule"},
            ],
            None,
            "user",
        ),
    ],
)
def test_system_messages(messages, expected_system, expected_first_role):
    req = chat.request_to_internal({"model": "gpt-4", "messages": messages})
    assert req.system == expected_system
    assert req.messages[0].role == expected_first_role


@pytest.mark.parametrize(
    "content,expected",
    [
        # String content becomes Text
        ("hello", [Text(text="hello")]),
        # Empty string becomes empty list
        ("", []),
        # Parts list
        (
            [{"type": "text", "text": "hello"}],
            [Text(text="hello")],
        ),
    ],
)
def test_user_content(content, expected):
    req = chat.request_to_internal(
        {"model": "gpt-4", "messages": [{"role": "user", "content": content}]}
    )
    assert req.messages[0].content == expected


@pytest.mark.parametrize(
    "content, shorthand", [("hi", True), ([{"type": "text", "text": "hi"}], False)]
)
def test_string_content_is_marked_shorthand(content, shorthand):
    """A bare string stays one when re-encoded as Anthropic Messages."""
    req = chat.request_to_internal(
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": content},
                {"role": "tool", "tool_call_id": "call_1", "content": content},
            ],
        }
    )
    user, assistant, tool_turn = req.messages
    flags = [user.content_shorthand, assistant.content_shorthand]
    assert flags + [tool_turn.content[0].content_shorthand] == [shorthand] * 3


def test_image_url_data_url():
    req = chat.request_to_internal(
        {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw0KG"},
                        }
                    ],
                }
            ],
        }
    )
    img = req.messages[0].content[0]
    assert isinstance(img, Image)
    assert isinstance(img.source, Base64Source)
    assert img.source.media_type == "image/png"
    assert img.source.data == "iVBORw0KG"


def test_image_url_https():
    req = chat.request_to_internal(
        {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/img.png", "detail": "high"},
                        }
                    ],
                }
            ],
        }
    )
    img = req.messages[0].content[0]
    assert isinstance(img, Image)
    assert isinstance(img.source, UrlSource)
    assert img.source.url == "https://example.com/img.png"
    assert img.extras.get("openai_chat") == {"detail": "high"}


@pytest.mark.parametrize(
    "reasoning_field,reasoning_value",
    [
        ("reasoning", "thinking text"),
        ("reasoning_content", "thinking text"),
    ],
)
def test_assistant_reasoning(reasoning_field, reasoning_value):
    req = chat.request_to_internal(
        {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", reasoning_field: reasoning_value, "content": "Hello"},
            ],
        }
    )
    blocks = req.messages[1].content
    assert isinstance(blocks[0], Thinking)
    assert blocks[0].thinking == reasoning_value
    assert isinstance(blocks[1], Text)


def test_assistant_tool_calls():
    req = chat.request_to_internal(
        {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                        }
                    ],
                }
            ],
        }
    )
    call = req.messages[0].content[0]
    assert isinstance(call, ToolCall)
    assert call.id == "call_1"
    assert call.name == "get_weather"
    assert call.input == {"city": "Paris"}
    assert call.raw_arguments is None


def test_assistant_tool_call_invalid_json():
    req = chat.request_to_internal(
        {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{bad json"},
                        }
                    ],
                }
            ],
        }
    )
    call = req.messages[0].content[0]
    assert call.input == {}
    assert call.raw_arguments == "{bad json"


def test_tool_messages_merge():
    req = chat.request_to_internal(
        {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "tool1", "arguments": "{}"},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "tool2", "arguments": "{}"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "result 1"},
                {"role": "tool", "tool_call_id": "call_2", "content": "result 2"},
            ],
        }
    )
    # Two consecutive tool messages merge into one user turn
    assert len(req.messages) == 2
    assert req.messages[1].role == "user"
    assert len(req.messages[1].content) == 2
    assert all(isinstance(b, ToolResult) for b in req.messages[1].content)
    assert req.messages[1].content[0].tool_call_id == "call_1"
    assert req.messages[1].content[1].tool_call_id == "call_2"


@pytest.mark.parametrize(
    "tool_choice,parallel,expected_mode,expected_name,expected_parallel",
    [
        ("auto", None, "auto", None, None),
        ("none", None, "none", None, None),
        ("required", None, "any", None, None),
        (None, True, "auto", None, True),
        (
            {"type": "function", "function": {"name": "my_tool"}},
            None,
            "tool",
            "my_tool",
            None,
        ),
    ],
)
def test_tool_choice(tool_choice, parallel, expected_mode, expected_name, expected_parallel):
    body = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if parallel is not None:
        body["parallel_tool_calls"] = parallel
    req = chat.request_to_internal(body)
    if expected_mode is None:
        assert req.tool_choice is None
    else:
        assert req.tool_choice.mode == expected_mode
        assert req.tool_choice.name == expected_name
        assert req.tool_choice.parallel == expected_parallel


def test_max_completion_tokens_preferred():
    req = chat.request_to_internal(
        {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_completion_tokens": 100,
            "max_tokens": 50,
        }
    )
    assert req.max_tokens == 100


def test_stop_string_to_list():
    req = chat.request_to_internal(
        {"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}], "stop": "STOP"}
    )
    assert req.stop == ["STOP"]


def test_reasoning_effort_none():
    req = chat.request_to_internal(
        {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "reasoning_effort": "none",
        }
    )
    assert req.thinking.mode == "disabled"


def test_response_format_json_schema():
    req = chat.request_to_internal(
        {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Person",
                    "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                    "strict": True,
                },
            },
        }
    )
    fmt = req.response_format
    assert fmt.type == "json_schema"
    assert fmt.name == "Person"
    assert fmt.json_schema["type"] == "object"
    assert fmt.strict is True


def test_unknown_keys_in_extras():
    req = chat.request_to_internal(
        {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "custom_field": "value",
        }
    )
    assert req.extras.get("openai_chat") == {"custom_field": "value"}


@pytest.mark.parametrize(
    "body,error_fragment",
    [
        ({"model": "gpt-4", "messages": [], "n": 2}, "n=2 is not supported"),
        ({"model": "gpt-4", "messages": [{"role": "unknown", "content": "Hi"}]}, "unknown"),
        ({"model": "gpt-4"}, "messages"),
        ({"model": "gpt-4", "messages": [], "functions": []}, "'functions' is deprecated"),
        (
            {
                "model": "gpt-4",
                "messages": [{"role": "assistant", "function_call": {"name": "f"}}],
            },
            "'function_call' is deprecated",
        ),
        (
            {
                "model": "gpt-4",
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "ok"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,xyz"},
                            },
                        ],
                    }
                ],
            },
            "text parts",
        ),
    ],
)
def test_request_errors(body, error_fragment):
    with pytest.raises(TranslationError, match=error_fragment):
        chat.request_to_internal(body)


# --- response_from_internal tests ---


def test_response_text_only():
    resp = Response(
        id="chatcmpl-123",
        model="gpt-4",
        content=[Text(text="Hello world")],
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=20),
    )
    out = chat.response_from_internal(resp, "gpt-4")
    validated = validate_chat_completion(out)
    assert validated.id == "chatcmpl-123"
    assert validated.model == "gpt-4"
    assert validated.choices[0].message.content == "Hello world"
    assert validated.choices[0].finish_reason == "stop"
    assert validated.usage.prompt_tokens == 10
    assert validated.usage.completion_tokens == 20


def test_response_tool_call():
    resp = Response(
        id="chatcmpl-123",
        model="gpt-4",
        content=[
            ToolCall(
                id="call_abc",
                name="get_weather",
                input={"location": "Paris"},
            )
        ],
        stop_reason="tool_use",
    )
    out = chat.response_from_internal(resp, "gpt-4")
    validated = validate_chat_completion(out)
    assert validated.choices[0].message.content is None
    assert validated.choices[0].finish_reason == "tool_calls"
    assert len(validated.choices[0].message.tool_calls) == 1
    call = validated.choices[0].message.tool_calls[0]
    assert call.id == "call_abc"
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"location": "Paris"}


def test_response_thinking():
    resp = Response(
        id="chatcmpl-123",
        model="gpt-4",
        content=[Thinking(thinking="Let me think..."), Text(text="Answer")],
        stop_reason="end_turn",
    )
    out = chat.response_from_internal(resp, "gpt-4")
    validated = validate_chat_completion(out)
    assert validated.choices[0].message.reasoning == "Let me think..."
    assert validated.choices[0].message.content == "Answer"


def test_response_usage_with_cache():
    resp = Response(
        id="chatcmpl-123",
        model="gpt-4",
        content=[Text(text="Hello")],
        stop_reason="end_turn",
        usage=Usage(input_tokens=100, output_tokens=20, cache_read_tokens=50),
    )
    out = chat.response_from_internal(resp, "gpt-4")
    validated = validate_chat_completion(out)
    assert validated.usage.prompt_tokens == 100
    assert validated.usage.prompt_tokens_details.cached_tokens == 50


def test_response_raw_arguments():
    resp = Response(
        id="chatcmpl-123",
        model="gpt-4",
        content=[
            ToolCall(
                id="call_abc",
                name="get_weather",
                input={},
                raw_arguments="{invalid json",
            )
        ],
        stop_reason="tool_use",
    )
    out = chat.response_from_internal(resp, "gpt-4")
    validated = validate_chat_completion(out)
    call = validated.choices[0].message.tool_calls[0]
    assert call.function.arguments == "{invalid json"
