"""Offline tests for the Ollama provider."""

import json

import httpx
import pytest

from model_calling.core import (
    Base64Source,
    Image,
    Message,
    Request,
    Text,
    Thinking,
    ThinkingConfig,
    Tool,
    ToolCall,
    ToolChoice,
    ToolResult,
    UrlSource,
)
from model_calling.formats import TranslationError, anthropic, chat_completions
from model_calling.providers.ollama import OllamaProvider
from tests.harness.replay import Exchange, ReplayTransport, fixture_path


def load_fixture(scenario: str) -> dict:
    return json.loads(fixture_path("ollama", scenario).read_text())


@pytest.fixture
def provider():
    return OllamaProvider(
        model="qwen3:4b",
        base_url="http://localhost:11434",
    )


# --- build_request tests ---


def test_build_request_system_message(provider):
    req = Request(
        model="qwen3:4b",
        messages=[Message(role="user", content=[Text(text="Hi")])],
        system=[Text(text="You are helpful.")],
    )
    body = provider.build_request(req)
    assert body["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert body["messages"][1] == {"role": "user", "content": "Hi"}


@pytest.mark.parametrize(
    "param,value,option_key",
    [
        ("max_tokens", 100, "num_predict"),
        ("temperature", 0.7, "temperature"),
        ("top_p", 0.9, "top_p"),
        ("top_k", 40, "top_k"),
        ("stop", ["STOP"], "stop"),
    ],
)
def test_build_request_options(provider, param, value, option_key):
    req = Request(
        model="qwen3:4b",
        messages=[Message(role="user", content=[Text(text="Hi")])],
        **{param: value},
    )
    body = provider.build_request(req)
    assert body["options"][option_key] == value


@pytest.mark.parametrize(
    "thinking_mode,expected_think",
    [
        ("enabled", True),
        ("disabled", False),
        ("adaptive", True),
    ],
)
def test_build_request_thinking(provider, thinking_mode, expected_think):
    req = Request(
        model="qwen3:4b",
        messages=[Message(role="user", content=[Text(text="Hi")])],
        thinking=ThinkingConfig(mode=thinking_mode),
    )
    body = provider.build_request(req)
    assert body["think"] == expected_think


@pytest.mark.parametrize(
    "response_format,expected_format",
    [
        ({"type": "json_object"}, "json"),
        ({"type": "json_schema", "json_schema": {"type": "object"}}, {"type": "object"}),
    ],
)
def test_build_request_response_format(provider, response_format, expected_format):
    from model_calling.core import ResponseFormat

    req = Request(
        model="qwen3:4b",
        messages=[Message(role="user", content=[Text(text="Hi")])],
        response_format=ResponseFormat(**response_format),
    )
    body = provider.build_request(req)
    assert body["format"] == expected_format


def test_build_request_assistant_tool_call(provider):
    call = ToolCall(id="call_1", name="get_weather", input={"location": "Paris"})
    req = Request(
        model="qwen3:4b",
        messages=[Message(role="assistant", content=[call])],
    )
    body = provider.build_request(req)
    assert body["messages"][0]["tool_calls"] == [
        {"function": {"name": "get_weather", "arguments": {"location": "Paris"}}}
    ]


def test_build_request_tool_result(provider):
    # First the assistant makes a tool call
    call = ToolCall(id="call_1", name="get_weather", input={"location": "Paris"})
    # Then user provides the result
    result = ToolResult(tool_call_id="call_1", content=[Text(text="Sunny, 20C")])
    req = Request(
        model="qwen3:4b",
        messages=[
            Message(role="assistant", content=[call]),
            Message(role="user", content=[result]),
        ],
    )
    body = provider.build_request(req)
    # Tool result becomes a tool message
    assert body["messages"][1] == {
        "role": "tool",
        "content": "Sunny, 20C",
        "tool_name": "get_weather",
    }
    # When user turn contains only tool results, no additional user message is emitted
    assert len(body["messages"]) == 2


def test_build_request_base64_image(provider):
    img = Image(source=Base64Source(media_type="image/png", data="iVBORw0KG"))
    req = Request(
        model="qwen3:4b",
        messages=[Message(role="user", content=[img])],
    )
    body = provider.build_request(req)
    assert body["messages"][0]["images"] == ["iVBORw0KG"]


def test_build_request_url_image_raises(provider):
    img = Image(source=UrlSource(url="https://example.com/img.png"))
    req = Request(
        model="qwen3:4b",
        messages=[Message(role="user", content=[img])],
    )
    with pytest.raises(TranslationError, match="base64 images"):
        provider.build_request(req)


def test_build_request_tool_choice_none_drops_tools(provider):
    req = Request(
        model="qwen3:4b",
        messages=[Message(role="user", content=[Text(text="Hi")])],
        tools=[Tool(name="get_weather", input_schema={"type": "object"})],
        tool_choice=ToolChoice(mode="none"),
    )
    body = provider.build_request(req)
    assert "tools" not in body


STRICT_WEATHER = {"name": "get_weather", "parameters": {"type": "object"}, "strict": True}
WEATHER = {"type": "function", "function": {"name": "get_weather"}}
HI = [{"role": "user", "content": "Hi"}]


@pytest.mark.parametrize(
    "to_internal, body, refused",
    [
        (chat_completions.request_to_internal, {"tool_choice": "required"}, ["tool_choice"]),
        (
            chat_completions.request_to_internal,
            {"tools": [WEATHER], "parallel_tool_calls": False},
            ["parallel_tool_calls"],
        ),
        # Without tools sent, one call per turn holds anyway.
        (chat_completions.request_to_internal, {"parallel_tool_calls": False}, []),
        (
            chat_completions.request_to_internal,
            {"tools": [WEATHER], "tool_choice": "none", "parallel_tool_calls": False},
            [],
        ),
        (
            chat_completions.request_to_internal,
            {"tools": [WEATHER], "parallel_tool_calls": True},
            [],
        ),
        (
            chat_completions.request_to_internal,
            {"tools": [{"type": "function", "function": STRICT_WEATHER}]},
            ["tools.strict"],
        ),
        (
            anthropic.request_to_internal,
            {
                "max_tokens": 16,
                "tools": [
                    {"name": "get_weather", "input_schema": {"type": "object"}, "strict": True}
                ],
                "tool_choice": {
                    "type": "tool",
                    "name": "get_weather",
                    "disable_parallel_tool_use": True,
                },
            },
            ["tool_choice", "parallel_tool_calls", "tools.strict"],
        ),
    ],
)
def test_unsupported_params(provider, to_internal, body, refused):
    """What strict mode refuses: forced tool calls, one call per turn, strict tools."""
    request = to_internal({"model": "qwen3:4b", "messages": HI, **body})
    assert provider.unsupported_params(request) == refused


# --- thinking levels ---

GPT_OSS_INFO = Exchange.load(fixture_path("ollama", "model_info-gpt_oss_20b"))
# qwen3:4b's `thinking` from /api/show (Ollama 0.40): on/off only.
QWEN3_INFO = {"thinking": {"values": [True], "default": True}}


def thinking_request(mode="enabled", effort="low") -> Request:
    return Request(
        model="m",
        messages=[Message(role="user", content=[Text(text="Hi")])],
        thinking=ThinkingConfig(mode=mode, effort=effort),
    )


@pytest.mark.parametrize(
    "info, mode, effort, think",
    [
        (GPT_OSS_INFO.response_json, "enabled", "low", "low"),
        (GPT_OSS_INFO.response_json, "adaptive", "high", "high"),
        # A level the model doesn't define, a model without levels, a failed
        # lookup: thinking on, at the model's own pace.
        (GPT_OSS_INFO.response_json, "enabled", "max", True),
        (QWEN3_INFO, "enabled", "low", True),
        (None, "enabled", "low", True),
        (GPT_OSS_INFO.response_json, "disabled", None, False),
    ],
)
def test_build_request_think(provider, info, mode, effort, think):
    provider.use_model_info(None if info is None else provider.read_model_info(info))
    assert provider.build_request(thinking_request(mode, effort))["think"] == think


@pytest.mark.parametrize(
    "request_, needed",
    [
        (thinking_request(), True),
        (thinking_request(effort=None), False),
        (thinking_request(mode="disabled", effort=None), False),
        (Request(model="m", messages=[]), False),
    ],
)
def test_needs_model_info_only_for_an_effort(provider, request_, needed):
    assert provider.needs_model_info(request_) is needed


def replayed_provider(exchanges, cache) -> tuple[OllamaProvider, ReplayTransport]:
    replay = ReplayTransport(exchanges)
    provider = OllamaProvider(
        model="gpt-oss:20b",
        base_url="http://localhost:11434",
        http_client=httpx.AsyncClient(transport=replay),
        model_info_cache=cache,
    )
    return provider, replay


async def test_prepare_looks_up_each_model_once():
    cache: dict = {}
    first, replay = replayed_provider([GPT_OSS_INFO], cache)
    await first.prepare(thinking_request())
    assert first.think_levels == ("low", "medium", "high")
    second, unused = replayed_provider([], cache)
    await second.prepare(thinking_request())
    assert second.think_levels == ("low", "medium", "high")
    assert len(replay.requests) == 1 and not unused.requests
    replay.check()


@pytest.mark.parametrize(
    "body", [{"thinking": True}, {"thinking": {"values": "low"}}, ["low"], {}, None]
)
def test_read_model_info_tolerates_any_shape(provider, body):
    assert provider.read_model_info(body) == ()


async def test_prepare_caches_only_the_levels():
    cache: dict = {}
    provider, replay = replayed_provider([GPT_OSS_INFO], cache)
    await provider.prepare(thinking_request())
    assert list(cache.values()) == [("low", "medium", "high")]
    replay.check()


async def test_prepare_falls_back_when_the_lookup_fails():
    url, body = OllamaProvider(
        model="gpt-oss:20b", base_url="http://localhost:11434"
    ).model_info_request()
    missing = Exchange("POST", url, body, 404, "application/json", {"error": "model not found"})
    cache: dict = {}
    provider, replay = replayed_provider([missing], cache)
    await provider.prepare(thinking_request())
    assert provider.build_request(thinking_request())["think"] is True
    assert not cache, "a failed lookup must not be cached"
    replay.check()


# --- parse_response tests ---


def test_parse_response_text(provider):
    fixture = load_fixture("text")
    body = fixture["response"]["json"]
    resp = provider.parse_response(body)

    assert isinstance(resp.content[0], Thinking)
    assert resp.content[0].thinking == body["message"]["thinking"]
    assert isinstance(resp.content[1], Text)
    assert resp.content[1].text == "pong"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == body["prompt_eval_count"]
    assert resp.usage.output_tokens == body["eval_count"]
    assert resp.usage.cache_read_tokens == body["prompt_eval_cached_count"]


def test_parse_response_tools(provider):
    fixture = load_fixture("tools")
    body = fixture["response"]["json"]
    resp = provider.parse_response(body)

    assert resp.stop_reason == "tool_use"
    call = next(b for b in resp.content if isinstance(b, ToolCall))
    assert call.id == body["message"]["tool_calls"][0]["id"]
    assert call.name == "get_weather"
    assert call.input == {"location": "Paris"}


def test_parse_response_unique_ids(provider):
    fixture = load_fixture("text")
    body = fixture["response"]["json"]
    resp1 = provider.parse_response(body)
    resp2 = provider.parse_response(body)
    assert resp1.id != resp2.id


# --- stream_parser tests ---


def test_stream_parser_basic(provider):
    fixture = load_fixture("stream")
    lines = fixture["response"]["lines"]

    parser = provider.stream_parser()
    all_events = []
    for line in lines:
        all_events.extend(parser.feed(line))
    all_events.extend(parser.close())

    from model_calling.core import (
        MessageStart,
        BlockStart,
        BlockDelta,
        MessageDelta,
        MessageStop,
    )

    # Check event order
    assert isinstance(all_events[0], MessageStart)
    assert all_events[0].model == "qwen3:4b"

    # Thinking block
    thinking_start = next(
        e for e in all_events if isinstance(e, BlockStart) and isinstance(e.block, Thinking)
    )
    assert thinking_start.index == 0

    # Text block
    text_start = next(
        e for e in all_events if isinstance(e, BlockStart) and isinstance(e.block, Text)
    )
    assert text_start.index == 1

    # Final events
    assert isinstance(all_events[-2], MessageDelta)
    assert isinstance(all_events[-1], MessageStop)

    # Concatenate text
    text_deltas = [e for e in all_events if isinstance(e, BlockDelta) and e.index == 1]
    text = "".join(d.delta.text for d in text_deltas)
    assert text == "pong"


def test_stream_parser_sequential_block_indexes(provider):
    fixture = load_fixture("stream")
    lines = fixture["response"]["lines"]

    parser = provider.stream_parser()
    all_events = []
    for line in lines:
        all_events.extend(parser.feed(line))

    from model_calling.core import BlockStart

    block_starts = [e for e in all_events if isinstance(e, BlockStart)]
    indexes = [e.index for e in block_starts]
    assert indexes == list(range(len(indexes)))


def test_stream_parser_truncated_stream_error(provider):
    # Only the first line, no done=true
    fixture = load_fixture("stream")
    lines = fixture["response"]["lines"][:1]

    parser = provider.stream_parser()
    for line in lines:
        parser.feed(line)

    from model_calling.core import StreamError

    close_events = parser.close()
    assert len(close_events) == 1
    assert isinstance(close_events[0], StreamError)
    assert "before it finished" in close_events[0].message


def test_stream_parser_error_line(provider):
    parser = provider.stream_parser()
    events = parser.feed('{"error": "model not found"}')

    from model_calling.core import StreamError

    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert "model not found" in events[0].message
