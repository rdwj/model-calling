"""
What each grid scenario must return through an OpenAI Responses ingress,
checked on the typed result: a response, or the list of stream events.

The library's own types and the `openai` SDK's have the same attributes,
so the library, gateway, and SDK-client cells share these checks, as
`checks.py` does for the other ingresses.
"""

import json

from tests.harness.checks import (
    EXPECTED_COLOR,
    EXPECTED_PARALLEL_TOOLS,
    EXPECTED_PARAMETERLESS_TOOL,
    EXPECTED_PRODUCT,
    EXPECTED_TEXT,
    EXPECTED_TOOL,
    _capital,
)
from tests.harness.sdk import validate_response, validate_responses_stream


def _text(resp) -> str:
    return "".join(
        part.text
        for item in resp.output
        if item.type == "message"
        for part in item.content
        if part.type == "output_text"
    )


def _valid(resp, model: str, status: str = "completed"):
    body = resp.to_dict()
    validate_response(body)
    assert resp.status == status, body
    assert resp.model == model
    # Nothing is stored, whatever the request asked.
    assert body["store"] is False
    assert resp.usage is not None and resp.usage.total_tokens > 0, resp.usage


def _response_text(resp, model: str, expected: str):
    _valid(resp, model)
    assert expected in _text(resp).lower(), resp.output


def response_text(resp, model: str):
    _response_text(resp, model, EXPECTED_TEXT)


def response_image(resp, model: str):
    _response_text(resp, model, EXPECTED_COLOR)


def response_reasoning(resp, model: str):
    _response_text(resp, model, EXPECTED_PRODUCT)
    kinds = [item.type for item in resp.output]
    assert kinds[0] == "reasoning", kinds
    assert resp.output[0].summary[0].text.strip(), "no reasoning returned"


def response_thinking_unsupported(resp, model: str):
    """Reasoning asked of a model that can't think: the drop path answers without it."""
    _response_text(resp, model, EXPECTED_PRODUCT)


def response_json_schema(resp, model: str):
    _valid(resp, model)
    _capital(_text(resp))


def _calls(resp, model: str) -> list:
    _valid(resp, model)
    calls = [(i.name, json.loads(i.arguments)) for i in resp.output if i.type == "function_call"]
    assert calls, f"no function_call items in {resp.output}"
    return calls


def response_tools(resp, model: str):
    assert _calls(resp, model)[0] == EXPECTED_TOOL


def response_parameterless_tool(resp, model: str):
    assert _calls(resp, model)[0] == EXPECTED_PARAMETERLESS_TOOL


def response_parallel_tools(resp, model: str):
    assert sorted(_calls(resp, model), key=str) == EXPECTED_PARALLEL_TOOLS


def response_parallel_tools_off(resp, model: str):
    calls = _calls(resp, model)
    assert len(calls) == 1 and calls[0] in EXPECTED_PARALLEL_TOOLS, calls


def response_tool_error(resp, model: str):
    """Told the call failed, the model either explains or retries."""
    _valid(resp, model)
    if not any(i.type == "function_call" for i in resp.output):
        assert _text(resp).strip(), resp.output


def _stream(events, model: str):
    """A well-formed event sequence, as the SDK's stream helper needs it:
    numbered from 0, created and in_progress first, every item added then
    done with its events between, and the whole response last. Returns
    that response."""
    validate_responses_stream([e.to_dict() for e in events])
    assert [e.sequence_number for e in events] == list(range(len(events)))
    kinds = [e.type for e in events]
    assert kinds[:2] == ["response.created", "response.in_progress"], kinds
    assert kinds[-1] == "response.completed", kinds
    lifecycle = [e.response for e in events if hasattr(e, "response")]
    assert len({r.id for r in lifecycle}) == 1 and {r.model for r in lifecycle} == {model}
    open_items: dict = {}
    for e in events:
        if e.type == "response.output_item.added":
            assert e.output_index not in open_items, kinds
            open_items[e.output_index] = e.item.id
        elif e.type == "response.output_item.done":
            assert open_items.pop(e.output_index) == e.item.id
        elif hasattr(e, "item_id"):
            assert open_items.get(e.output_index) == e.item_id, f"{e.type} outside its item"
    assert not open_items, f"items never done: {open_items}"
    final = events[-1].response
    done = [e.item.to_dict() for e in events if e.type == "response.output_item.done"]
    assert [i.to_dict() for i in final.output] == done
    return final


def response_stream(events, model: str):
    final = _stream(events, model)
    text = "".join(e.delta for e in events if e.type == "response.output_text.delta")
    assert EXPECTED_TEXT in text.lower()
    response_text(final, model)


def response_stream_tools(events, model: str):
    final = _stream(events, model)
    calls = [
        e.item
        for e in events
        if e.type == "response.output_item.added" and e.item.type == "function_call"
    ]
    assert calls, "no function_call item"
    arguments = "".join(
        e.delta
        for e in events
        if e.type == "response.function_call_arguments.delta" and e.item_id == calls[0].id
    )
    assert (calls[0].name, json.loads(arguments)) == EXPECTED_TOOL
    response_tools(final, model)


RESPONSE_CHECKS = {
    "text": response_text,
    "store": response_text,
    "tools": response_tools,
    "parallel_tools": response_parallel_tools,
    "tool_error": response_tool_error,
    "image": response_image,
    "json_schema": response_json_schema,
    "stream": response_stream,
    "stream_tools": response_stream_tools,
    "reasoning_effort": response_reasoning,
    "thinking_unsupported": response_thinking_unsupported,
    "tool_required": response_tools,
    "tool_named": response_tools,
    "tool_strict": response_tools,
    "tool_parameterless": response_parameterless_tool,
    "parallel_tools_off": response_parallel_tools_off,
}


def _assembled(resp, model: str):
    """A response the SDK assembled from a stream, already its own type."""
    assert resp.status == "completed" and resp.model == model, resp
    assert resp.usage is not None and resp.usage.total_tokens > 0


def assembled_text(resp, model: str):
    _assembled(resp, model)
    assert EXPECTED_TEXT in resp.output_text.lower(), resp.output


def assembled_tools(resp, model: str):
    _assembled(resp, model)
    calls = [(i.name, json.loads(i.arguments)) for i in resp.output if i.type == "function_call"]
    assert calls and calls[0] == EXPECTED_TOOL, resp.output


# What the SDK's stream helper must assemble from each stream scenario.
FINAL_RESPONSE_CHECKS = {"stream": assembled_text, "stream_tools": assembled_tools}
