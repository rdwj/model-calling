"""Offline tests for the Cohere v2 provider, beyond what the grid replays."""

import json

import pytest

from model_calling.core import (
    Base64Source,
    BlockDelta,
    BlockStart,
    BlockStop,
    Document,
    FileSource,
    Image,
    Message,
    MessageDelta,
    MessageStart,
    MessageStop,
    Request,
    ResponseFormat,
    StreamError,
    Text,
    Thinking,
    ThinkingConfig,
    Tool,
    ToolCall,
    ToolChoice,
    ToolResult,
    UrlSource,
)
from model_calling.core.embeddings import EmbeddingRequest
from model_calling.errors import APIError
from model_calling.formats import TranslationError
from model_calling.formats.embeddings import FORMAT as EMBEDDINGS_FORMAT
from model_calling.providers import default_configs, provider_for
from model_calling.providers.cohere import CohereProvider

MODEL = "command-a-03-2025"
USER = Message(role="user", content=[Text(text="Hi")], content_shorthand=True)
WEATHER = Tool(name="get_weather", input_schema={"type": "object", "properties": {}})
TIME = Tool(name="get_time", input_schema={"type": "object", "properties": {}})


@pytest.fixture
def provider():
    return CohereProvider(model=MODEL, base_url="https://api.cohere.com/v2")


def request(**fields) -> Request:
    return Request(**{"model": MODEL, "messages": [USER], **fields})


def test_default_config_reaches_v2_with_the_key_from_the_environment():
    configs = default_configs({"COHERE_API_KEY": "k"})
    cohere = provider_for(f"cohere/{MODEL}", configs)
    assert cohere.endpoint() == "https://api.cohere.com/v2/chat"
    assert cohere.embedding_endpoint() == "https://api.cohere.com/v2/embed"
    assert cohere.headers() == {"authorization": "Bearer k"}


# --- request ------------------------------------------------------------------


@pytest.mark.parametrize(
    "system, content",
    [
        ([Text(text="Be terse.")], "Be terse."),
        (
            [Text(text="Be terse."), Text(text="Use lowercase.")],
            [{"type": "text", "text": "Be terse."}, {"type": "text", "text": "Use lowercase."}],
        ),
    ],
    ids=["one_block", "two_blocks"],
)
def test_system_prompt_is_the_first_message(provider, system, content):
    body = provider.build_request(request(system=system))
    assert "system" not in body
    assert body["messages"][0] == {"role": "system", "content": content}


def test_history_keeps_roles_reasoning_and_calls(provider):
    call = ToolCall(id="c1", name="get_weather", input={"location": "Paris"})
    broken = ToolCall(id="c2", name="get_weather", input={}, raw_arguments='{"loc')
    messages = [
        USER,
        Message(
            role="assistant",
            content=[Thinking(thinking="Look it up.", signature="sig"), call, broken],
        ),
        Message(
            role="user",
            content=[
                ToolResult(tool_call_id="c1", content=[Text(text="sunny")], is_error=False),
                ToolResult(tool_call_id="c2", content=[Text(text="bad")], content_shorthand=True),
            ],
        ),
        Message(role="system", content=[Text(text="Answer briefly.")], content_shorthand=True),
    ]
    assert provider.build_request(request(messages=messages))["messages"][1:] == [
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "Look it up."}],
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"loc'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "sunny"}]},
        {"role": "tool", "tool_call_id": "c2", "content": "bad"},
        {"role": "system", "content": "Answer briefly."},
    ]


@pytest.mark.parametrize(
    "source, url",
    [
        (Base64Source(media_type="image/png", data="AAAA"), "data:image/png;base64,AAAA"),
        (UrlSource(url="https://example.test/a.png"), "https://example.test/a.png"),
    ],
)
def test_images_are_image_url_blocks(provider, source, url):
    msg = Message(role="user", content=[Image(source=source), Text(text="What is it?")])
    assert provider.build_request(request(messages=[msg]))["messages"][0]["content"] == [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "What is it?"},
    ]


@pytest.mark.parametrize(
    "block, match",
    [
        (Image(source=FileSource(file_id="f")), "image by file id"),
        (Document(data={"source": {}}), "document blocks"),
    ],
)
def test_untranslatable_blocks_are_refused(provider, block, match):
    with pytest.raises(TranslationError, match=match):
        provider.build_request(request(messages=[Message(role="user", content=[block])]))


def test_sampling_parameters_use_v2_names(provider):
    body = provider.build_request(
        request(
            max_tokens=9,
            temperature=0.2,
            top_p=0.5,
            top_k=40,
            seed=7,
            presence_penalty=0.1,
            frequency_penalty=0.3,
            stop=["###"],
        )
    )
    assert {k: v for k, v in body.items() if k not in ("model", "messages")} == {
        "max_tokens": 9,
        "temperature": 0.2,
        "p": 0.5,
        "k": 40,
        "seed": 7,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.3,
        "stop_sequences": ["###"],
    }


def test_more_stop_sequences_than_v2_takes_are_refused(provider):
    with pytest.raises(TranslationError, match="at most 5 stop sequences, not 6"):
        provider.build_request(request(stop=list("abcdef")))


@pytest.mark.parametrize(
    "fields, sent",
    [
        ({"thinking": ThinkingConfig(mode="disabled")}, {"thinking": {"type": "disabled"}}),
        ({"thinking": ThinkingConfig(mode="adaptive")}, {"thinking": {"type": "enabled"}}),
        (
            {"thinking": ThinkingConfig(mode="enabled", budget_tokens=512, effort="low")},
            {"thinking": {"type": "enabled", "token_budget": 512}},
        ),
        (
            {"tools": [WEATHER], "tool_choice": ToolChoice(mode="any")},
            {"tool_choice": "REQUIRED"},
        ),
        ({"tools": [WEATHER], "tool_choice": ToolChoice(mode="none")}, {"tool_choice": "NONE"}),
        ({"tools": [WEATHER], "tool_choice": ToolChoice(mode="auto")}, {}),
        (
            {"tools": [WEATHER.model_copy(update={"strict": True})]},
            {"strict_tools": True},
        ),
        (
            {"response_format": ResponseFormat(type="json_object")},
            {"response_format": {"type": "json_object"}},
        ),
        (
            {"response_format": ResponseFormat(type="json_schema", name="x", json_schema={})},
            {"response_format": {"type": "json_object", "json_schema": {}}},
        ),
    ],
    ids=[
        "thinking_disabled",
        "thinking_adaptive",
        "thinking_budget",
        "tool_choice_any",
        "tool_choice_none",
        "tool_choice_auto",
        "strict_tools",
        "json_object",
        "json_schema",
    ],
)
def test_supported_options(provider, fields, sent):
    req = request(**fields)
    assert provider.unsupported_params(req) == []
    body = provider.build_request(req)
    assert {k: body[k] for k in sent} == sent
    assert not {"tool_choice", "strict_tools"} & (set(body) - set(sent))


@pytest.mark.parametrize(
    "fields, refused, dropped",
    [
        (
            {"tools": [WEATHER, TIME], "tool_choice": ToolChoice(mode="tool", name="get_time")},
            ["tool_choice"],
            "tool_choice",
        ),
        (
            {"tools": [WEATHER], "tool_choice": ToolChoice(mode="auto", parallel=False)},
            ["parallel_tool_calls"],
            "parallel_tool_calls",
        ),
        (
            {"tools": [WEATHER.model_copy(update={"strict": True}), TIME]},
            ["tools.strict"],
            "strict_tools",
        ),
        ({"thinking": ThinkingConfig(mode="adaptive", effort="high")}, ["thinking.effort"], None),
        (
            {"tools": [WEATHER], "response_format": ResponseFormat(type="json_object")},
            ["response_format"],
            "response_format",
        ),
    ],
    ids=["named_tool", "parallel_off", "mixed_strict", "effort_only", "format_with_tools"],
)
def test_what_v2_cannot_express_is_refused_and_dropped(provider, fields, refused, dropped):
    req = request(**fields)
    assert provider.unsupported_params(req) == refused
    body = provider.build_request(req)
    if dropped:
        assert dropped not in body
    else:
        assert body["thinking"] == {"type": "enabled"}


# --- response -----------------------------------------------------------------


def reply(message: dict, finish: str = "COMPLETE", usage: dict | None = None) -> dict:
    body = {"id": "r1", "finish_reason": finish, "message": {"role": "assistant", **message}}
    if usage is not None:
        body["usage"] = usage
    return body


def test_response_keeps_reasoning_plan_and_calls_in_order(provider):
    body = reply(
        {
            "content": [{"type": "thinking", "thinking": "Need the tool."}],
            "tool_plan": "I will look it up.",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{"},
                },
            ],
        },
        finish="TOOL_CALL",
    )
    resp = provider.parse_response(body)
    assert resp.content == [
        Thinking(thinking="Need the tool."),
        Text(text="I will look it up."),
        ToolCall(id="c1", name="get_weather", input={"location": "Paris"}),
        ToolCall(id="c2", name="get_weather", input={}, raw_arguments="{"),
    ]
    assert (resp.id, resp.model, resp.stop_reason) == ("r1", MODEL, "tool_use")


@pytest.mark.parametrize(
    "usage, expected",
    [
        (
            {
                "billed_units": {"input_tokens": 5},
                "tokens": {"input_tokens": 71.0, "output_tokens": 3.0},
            },
            (71, 3, None),
        ),
        ({"billed_units": {"input_tokens": 5, "output_tokens": 3}}, (5, 3, None)),
        ({"tokens": {"input_tokens": 9, "output_tokens": 1}, "cached_tokens": 4.0}, (9, 1, 4)),
    ],
    ids=["tokens_first", "billed_fallback", "cached"],
)
def test_usage(provider, usage, expected):
    got = provider.parse_response(reply({"content": []}, usage=usage)).usage
    assert (got.input_tokens, got.output_tokens, got.cache_read_tokens) == expected


@pytest.mark.parametrize("finish", ["ERROR", "TIMEOUT"])
def test_a_failed_generation_raises(provider, finish):
    with pytest.raises(APIError, match=f"failed \\({finish}\\)"):
        provider.parse_response(reply({"content": []}, finish=finish))


# --- stream -------------------------------------------------------------------


def feed(provider, *events: dict, close: bool = True) -> list:
    parser = provider.stream_parser()
    out = []
    for event in events:
        out += parser.feed(f"event: {event['type']}")
        out += parser.feed("data: " + json.dumps(event))
    return out + (parser.close() if close else [])


START = {"type": "message-start", "id": "m1", "delta": {"message": {"role": "assistant"}}}


def content(kind: str, index: int, value: str) -> list:
    return [
        {
            "type": "content-start",
            "index": index,
            "delta": {"message": {"content": {"type": kind, kind: ""}}},
        },
        {"type": "content-delta", "index": index, "delta": {"message": {"content": {kind: value}}}},
        {"type": "content-end", "index": index},
    ]


def tool_call(index: int, call_id: str, arguments: str) -> list:
    start = {"id": call_id, "type": "function", "function": {"name": "f", "arguments": ""}}
    return [
        {"type": "tool-call-start", "index": index, "delta": {"message": {"tool_calls": start}}},
        {
            "type": "tool-call-delta",
            "index": index,
            "delta": {"message": {"tool_calls": {"function": {"arguments": arguments}}}},
        },
        {"type": "tool-call-end", "index": index},
    ]


def end(finish: str = "COMPLETE", **extra) -> dict:
    usage = {"tokens": {"input_tokens": 10, "output_tokens": 2}}
    return {"type": "message-end", "delta": {"finish_reason": finish, "usage": usage, **extra}}


def test_stream_numbers_content_and_tool_calls_in_one_sequence(provider):
    """Cohere counts content blocks and tool calls from 0 each; internal
    indexes must not collide."""
    events = feed(
        provider,
        START,
        *content("thinking", 0, "hmm"),
        *content("text", 1, "Calling."),
        *tool_call(0, "c1", "{}"),
        *tool_call(1, "c2", '{"a": 1}'),
        end("TOOL_CALL"),
    )
    starts = [(e.index, e.block.type) for e in events if isinstance(e, BlockStart)]
    assert starts == [(0, "thinking"), (1, "text"), (2, "tool_call"), (3, "tool_call")]
    assert [e.index for e in events if isinstance(e, BlockStop)] == [0, 1, 2, 3]
    assert events[0] == MessageStart(id="m1", model=MODEL)
    assert isinstance(events[-1], MessageStop)
    delta = events[-2]
    assert isinstance(delta, MessageDelta) and delta.stop_reason == "tool_use"
    assert (delta.usage.input_tokens, delta.usage.output_tokens) == (10, 2)


def test_stream_tool_plan_is_a_text_block_closed_by_the_first_call(provider):
    plan = [
        {"type": "tool-plan-delta", "delta": {"message": {"tool_plan": t}}} for t in ("I ", "will")
    ]
    events = feed(provider, START, *plan, *tool_call(0, "c1", "{}"), end("TOOL_CALL"))
    blocks = [e for e in events if isinstance(e, (BlockStart, BlockDelta, BlockStop))]
    assert [(type(e).__name__, e.index) for e in blocks] == [
        ("BlockStart", 0),
        ("BlockDelta", 0),
        ("BlockDelta", 0),
        ("BlockStop", 0),
        ("BlockStart", 1),
        ("BlockDelta", 1),
        ("BlockStop", 1),
    ]
    assert blocks[0].block == Text(text="")
    assert "".join(e.delta.text for e in blocks[1:3]) == "I will"


@pytest.mark.parametrize(
    "events, message",
    [
        ([START, *content("text", 0, "po")[1:2]], "content-delta for unopened block 0"),
        ([START, *tool_call(0, "c", "{}")[1:2]], "tool-call-delta for unopened call 0"),
        ([START, end("ERROR", error="overloaded")], "failed (ERROR): overloaded"),
        ([START, end("TIMEOUT")], "failed (TIMEOUT)"),
        ([START, *content("text", 0, "po")], "closed the stream before message-end"),
    ],
    ids=["delta_before_start", "call_delta_before_start", "error", "timeout", "cut_off"],
)
def test_stream_failures_end_in_an_error(provider, events, message):
    # The stream ends at its first error.
    errors = [e for e in feed(provider, *events) if isinstance(e, StreamError)]
    assert errors and message in errors[0].message, errors


def test_stream_non_json_data_is_an_error(provider):
    parser = provider.stream_parser()
    (error,) = parser.feed("data: {broken")
    assert isinstance(error, StreamError) and "non-JSON" in error.message


# --- embeddings ---------------------------------------------------------------


def embedding(**fields) -> EmbeddingRequest:
    return EmbeddingRequest(**{"model": "embed-v4.0", "input": ["a", "b"], **fields})


def test_embedding_request_takes_input_type_and_dimensions(provider):
    req = embedding(dimensions=256, extras={EMBEDDINGS_FORMAT: {"input_type": "search_query"}})
    assert provider.unsupported_embedding_params(req) == []
    assert provider.build_embedding_request(req) == {
        "model": MODEL,
        "texts": ["a", "b"],
        "input_type": "search_query",
        "embedding_types": ["float"],
        "output_dimension": 256,
    }


def test_embedding_request_refuses_other_extras_and_token_arrays(provider):
    req = embedding(extras={EMBEDDINGS_FORMAT: {"truncate": "END", "input_type": "clustering"}})
    assert provider.unsupported_embedding_params(req) == ["truncate"]
    with pytest.raises(TranslationError, match="text only"):
        provider.build_embedding_request(embedding(input=[[1, 2]]))


def test_embedding_response_reads_float_vectors_and_meta(provider):
    body = {
        "id": "e1",
        "embeddings": {"float": [[0.1], [0.2]]},
        "meta": {"billed_units": {"input_tokens": 4.0}},
    }
    resp = provider.parse_embedding_response(body)
    assert (resp.vectors, resp.input_tokens, resp.model) == ([[0.1], [0.2]], 4, MODEL)
