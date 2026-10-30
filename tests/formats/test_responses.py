"""OpenAI Responses requests to the internal format, and responses back."""

import json

import pytest

from model_calling.core import (
    Base64Source,
    FileSource,
    Image,
    Opaque,
    RedactedThinking,
    Response,
    Text,
    Thinking,
    ThinkingConfig,
    ToolCall,
    ToolChoice,
    ToolResult,
    Usage,
    UrlSource,
)
from model_calling.formats import TranslationError
from model_calling.formats import responses
from tests.harness.sdk import validate_response, validate_responses_request

MODEL = "m"


def request(**fields):
    body = {"model": MODEL, "input": "hi", **fields}
    return responses.request_to_internal(body)


# --- stateless and scope rules ----------------------------------------------


@pytest.mark.parametrize(
    "fields, named",
    [
        ({"previous_response_id": "resp_1"}, "previous_response_id"),
        ({"conversation": "conv_1"}, "conversation"),
        ({"conversation": {"id": "conv_1"}}, "conversation"),
        ({"background": True}, "background"),
        ({"prompt": {"id": "pmpt_1"}}, "prompt"),
        ({"include": ["reasoning.encrypted_content"]}, "reasoning.encrypted_content"),
        ({"tools": [{"type": "web_search"}]}, "'web_search'"),
        ({"tools": [{"type": "web_search_preview"}]}, "'web_search_preview'"),
        ({"tools": [{"type": "file_search", "vector_store_ids": []}]}, "'file_search'"),
        ({"tools": [{"type": "code_interpreter", "container": "c"}]}, "'code_interpreter'"),
        ({"tools": [{"type": "computer"}]}, "computer use"),
        ({"tools": [{"type": "computer_use_preview"}]}, "computer use"),
        ({"tools": [{"type": "mcp", "server_label": "s"}]}, "'mcp'"),
        ({"tool_choice": {"type": "web_search_preview"}}, "'web_search_preview'"),
        ({"input": [{"type": "item_reference", "id": "msg_1"}]}, "'item_reference'"),
        (
            {"input": [{"type": "reasoning", "id": "rs", "summary": [], "encrypted_content": "x"}]},
            "reasoning.encrypted_content",
        ),
        (
            {"input": [{"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}]},
            "'input_file'",
        ),
        ({"input": [{"type": "function_call_output", "output": "x"}]}, "call_id"),
        ({"instructions": [{"role": "system", "content": "x"}]}, "instructions"),
    ],
)
def test_refuses_what_needs_state_or_is_out_of_scope(fields, named):
    with pytest.raises(TranslationError, match="not supported|needs|must be") as err:
        request(**fields)
    assert named in str(err.value)


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"store": True},
        {"store": False},
        {"previous_response_id": None, "conversation": None, "background": False},
        {"include": ["message.output_text.logprobs"]},
    ],
)
def test_accepts_what_needs_no_state(fields):
    req = request(**fields)
    assert req.messages[0].content == [Text(text="hi")]
    assert "store" not in req.extras_for(responses.FORMAT)


def test_missing_input_is_refused():
    with pytest.raises(TranslationError, match="input is required"):
        responses.request_to_internal({"model": MODEL})


# --- requests -----------------------------------------------------------------

IMAGE_URL = "data:image/png;base64,QUJD"


def test_instructions_and_leading_system_messages_are_the_system_prompt():
    req = request(
        instructions="Be terse.",
        input=[
            {"role": "developer", "content": "Use lowercase."},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": [{"type": "input_text", "text": "Later."}]},
        ],
    )
    assert [b.text for b in req.system] == ["Be terse.", "Use lowercase."]
    assert req.system_shorthand
    assert [m.role for m in req.messages] == ["user", "system"]
    assert req.messages[1].content == [Text(text="Later.")]


def test_input_items_become_turns():
    req = request(
        input=[
            {"role": "user", "content": "weather?"},
            {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "a"}]},
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": '{"x": 1}'},
            {"type": "function_call", "call_id": "c2", "name": "f", "arguments": "not json"},
            {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
            {
                "type": "function_call_output",
                "call_id": "c2",
                "output": [{"type": "input_text", "text": "rain"}],
            },
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Sunny.", "annotations": []}],
            },
        ]
    )
    user, assistant, results, reply = req.messages
    assert user.content_shorthand and user.content == [Text(text="weather?")]
    thinking, first, second = assistant.content
    assert thinking == Thinking(thinking="a")
    assert (first.id, first.input) == ("c1", {"x": 1})
    assert (second.input, second.raw_arguments) == ({}, "not json")
    assert [r.tool_call_id for r in results.content] == ["c1", "c2"]
    assert results.content[0].content_shorthand and not results.content[1].content_shorthand
    assert reply.role == "assistant" and reply.content[0].text == "Sunny."
    assert reply.extras_for(responses.FORMAT) == {"id": "msg_1", "status": "completed"}


@pytest.mark.parametrize(
    "part, source",
    [
        ({"image_url": IMAGE_URL}, Base64Source(media_type="image/png", data="QUJD")),
        ({"image_url": "https://x/y.png"}, UrlSource(url="https://x/y.png")),
        ({"file_id": "file_1"}, FileSource(file_id="file_1")),
    ],
)
def test_input_images(part, source):
    content = [{"type": "input_image", "detail": "high", **part}]
    image = request(input=[{"role": "user", "content": content}]).messages[0].content[0]
    assert image == Image(source=source, extras={responses.FORMAT: {"detail": "high"}})


def test_tools_and_settings():
    req = request(
        tools=[
            {"type": "function", "name": "f", "parameters": None, "strict": None},
            {"type": "function", "name": "g", "parameters": {"type": "object"}, "strict": True},
        ],
        tool_choice={"type": "function", "name": "g"},
        parallel_tool_calls=False,
        max_output_tokens=64,
        temperature=0.5,
        top_p=0.9,
        reasoning={"effort": "low", "summary": "auto"},
        text={"format": {"type": "json_schema", "name": "x", "schema": {}}, "verbosity": "low"},
        metadata={"k": "v"},
    )
    f, g = req.tools
    assert f.input_schema == responses.EMPTY_SCHEMA and f.strict is None
    assert g.strict is True
    assert req.tool_choice == ToolChoice(mode="tool", name="g", parallel=False)
    assert (req.max_tokens, req.temperature, req.top_p) == (64, 0.5, 0.9)
    assert req.thinking.effort == "low" and req.thinking.mode == "enabled"
    assert (req.response_format.type, req.response_format.name) == ("json_schema", "x")
    assert req.extras_for(responses.FORMAT) == {
        "text": {"verbosity": "low"},
        "metadata": {"k": "v"},
    }


@pytest.mark.parametrize("choice, mode", [("auto", "auto"), ("none", "none"), ("required", "any")])
def test_tool_choice_modes(choice, mode):
    assert request(tool_choice=choice).tool_choice.mode == mode


def test_reasoning_effort_none_disables_thinking():
    assert request(reasoning={"effort": "none"}).thinking.mode == "disabled"


# --- responses ----------------------------------------------------------------


def test_response_items_echo_and_usage():
    body = {
        "model": MODEL,
        "input": "hi",
        "instructions": "Be terse.",
        "tools": [{"type": "function", "name": "f", "parameters": None, "strict": None}],
        "tool_choice": "required",
        "store": True,
    }
    validate_responses_request(body)
    req = responses.request_to_internal(body)
    resp = Response(
        id="r1",
        model="backend",
        content=[
            Thinking(thinking="hmm", signature="sig"),
            RedactedThinking(data="x"),
            Text(text="pong"),
            ToolCall(id="c1", name="f", input={"a": 1}),
        ],
        stop_reason="tool_use",
        usage=Usage(input_tokens=5, output_tokens=7, cache_read_tokens=2, reasoning_tokens=3),
    )
    out = responses.response_from_internal(resp, "alias", req)
    sdk = validate_response(out)
    assert [i.type for i in sdk.output] == ["reasoning", "message", "function_call"]
    assert [i.id for i in sdk.output] == ["rs_r1_0", "msg_r1_1", "fc_r1_2"]
    assert sdk.output[0].summary[0].text == "hmm"
    assert sdk.output_text == "pong"
    assert json.loads(sdk.output[2].arguments) == {"a": 1} and sdk.output[2].call_id == "c1"
    assert (sdk.status, sdk.model, sdk.instructions, out["store"]) == (
        "completed",
        "alias",
        "Be terse.",
        False,
    )
    assert sdk.tool_choice == "required" and sdk.tools[0].name == "f"
    assert (sdk.usage.input_tokens, sdk.usage.total_tokens) == (5, 12)
    assert sdk.usage.input_tokens_details.cached_tokens == 2
    assert sdk.usage.output_tokens_details.reasoning_tokens == 3


@pytest.mark.parametrize(
    "stop_reason, reason",
    [("max_tokens", "max_output_tokens"), ("refusal", "content_filter")],
)
def test_response_cut_short_is_incomplete(stop_reason, reason):
    resp = Response(id="r", model=MODEL, content=[Text(text="po")], stop_reason=stop_reason)
    sdk = validate_response(responses.response_from_internal(resp, MODEL, request()))
    assert sdk.status == "incomplete" and sdk.incomplete_details.reason == reason
    assert sdk.output[-1].status == "incomplete"
    # No usage reported is zeros, since Responses requires every count.
    assert sdk.usage.total_tokens == 0


@pytest.mark.parametrize(
    "block",
    [Opaque(format="anthropic", data={"type": "server_tool_use"}), ToolResult(tool_call_id="c")],
)
def test_response_block_with_no_item_is_an_error(block):
    resp = Response(id="r", model=MODEL, content=[block])
    with pytest.raises(TranslationError, match="no Responses output item"):
        responses.response_from_internal(resp, MODEL, request())


def test_reasoning_echo():
    req = request(reasoning={"effort": "high"})
    assert req.thinking == ThinkingConfig(mode="enabled", effort="high")
    resp = Response(id="r", model=MODEL, content=[])
    assert responses.response_from_internal(resp, MODEL, req)["reasoning"] == {"effort": "high"}
