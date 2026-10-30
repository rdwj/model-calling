"""Internal stream events to OpenAI Responses stream events."""

import pytest
from openai import omit
from openai.lib.streaming.responses import ResponseStreamState

from model_calling.core import (
    BlockDelta,
    BlockStart,
    BlockStop,
    MessageDelta,
    MessageStart,
    MessageStop,
    RedactedThinking,
    SignatureDelta,
    StreamError,
    Text,
    TextDelta,
    Thinking,
    ThinkingDelta,
    ToolCall,
    ToolInputDelta,
    Usage,
)
from model_calling.formats import responses
from model_calling.formats.responses_stream import ResponsesStreamEncoder
from model_calling.types.responses import Response, parse_stream_event
from tests.harness.sdk import validate_responses_stream

REQUEST = responses.request_to_internal({"model": "m", "input": "hi"})


def encode(events, request=REQUEST):
    encoder = ResponsesStreamEncoder("alias", request)
    return [out for event in events for out in encoder.encode(event)]


def assemble(payloads):
    """Run the payloads through the state the SDK's stream helper keeps,
    which raises on an event out of order, and return the final response."""
    state = ResponseStreamState(input_tools=omit, text_format=omit)
    fired = []
    for event in validate_responses_stream(payloads):
        fired += state.handle_event(event)
    return fired[-1].response


def blocks(*items):
    """Start, deltas, and stop events for (index, block, deltas) items."""
    events = []
    for index, block, deltas in items:
        events.append(BlockStart(index=index, block=block))
        events += [BlockDelta(index=index, delta=d) for d in deltas]
        events.append(BlockStop(index=index))
    return events


def stream(*items, stop_reason="end_turn", usage=None):
    return [
        MessageStart(id="r1", model="backend"),
        *blocks(*items),
        MessageDelta(stop_reason=stop_reason, usage=usage),
        MessageStop(),
    ]


def test_thinking_text_and_tool_call():
    payloads = encode(
        stream(
            (
                0,
                Thinking(thinking=""),
                [ThinkingDelta(thinking="hm"), SignatureDelta(signature="s")],
            ),
            (1, Text(text=""), [TextDelta(text="po"), TextDelta(text="ng")]),
            (2, ToolCall(id="c1", name="f", input={}), [ToolInputDelta(partial_json='{"a":')]),
            (3, ToolCall(id="c2", name="g", input={}), [ToolInputDelta(partial_json="1}")]),
            stop_reason="tool_use",
            usage=Usage(input_tokens=3, output_tokens=4),
        )
    )
    assert [p["sequence_number"] for p in payloads] == list(range(len(payloads)))
    assert payloads[0]["type"] == "response.created"
    assert payloads[0]["response"]["status"] == "in_progress"
    final = assemble(payloads)
    assert final.status == "completed" and final.model == "alias"
    assert [i.id for i in final.output] == ["rs_r1_0", "msg_r1_1", "fc_r1_2", "fc_r1_3"]
    assert final.output_text == "pong" and final.output[0].summary[0].text == "hm"
    assert [i.arguments for i in final.output[2:]] == ['{"a":', "1}"]
    assert final.usage.total_tokens == 7
    # Every payload is also one of the library's own event types.
    completed = parse_stream_event(payloads[-1])
    assert isinstance(completed.response, Response)
    assert completed.response.output_text == "pong"


def test_tool_call_with_no_arguments_streamed_gets_an_empty_object():
    payloads = encode(stream((0, ToolCall(id="c", name="f", input={}), []), stop_reason="tool_use"))
    deltas = [p["delta"] for p in payloads if p["type"].endswith("arguments.delta")]
    assert deltas == ["{}"]
    assert assemble(payloads).output[0].arguments == "{}"


def test_cut_short_stream_ends_incomplete():
    payloads = encode(stream((0, Text(text=""), [TextDelta(text="po")]), stop_reason="max_tokens"))
    assert payloads[-1]["type"] == "response.incomplete"
    response = payloads[-1]["response"]
    assert response["incomplete_details"] == {"reason": "max_output_tokens"}
    assert response["output"][-1]["status"] == "incomplete"
    # The item's own done event said it completed; only the response marks it cut short.
    done = [p for p in payloads if p["type"] == "response.output_item.done"]
    assert done[-1]["item"]["status"] == "completed"
    validate_responses_stream(payloads)


def test_redacted_thinking_has_no_item():
    payloads = encode(
        stream((0, RedactedThinking(data="x"), []), (1, Text(text=""), [TextDelta(text="hi")]))
    )
    final = assemble(payloads)
    assert [i.id for i in final.output] == ["msg_r1_0"]


def test_items_finishing_out_of_order_keep_their_output_order():
    events = [
        MessageStart(id="r1", model="m"),
        BlockStart(index=0, block=ToolCall(id="a", name="f", input={})),
        BlockStart(index=1, block=ToolCall(id="b", name="f", input={})),
        BlockStop(index=1),
        BlockStop(index=0),
        MessageDelta(stop_reason="tool_use"),
        MessageStop(),
    ]
    payloads = encode(events)
    assert [i["call_id"] for i in payloads[-1]["response"]["output"]] == ["a", "b"]
    validate_responses_stream(payloads)


def test_stream_error_is_an_error_event():
    payloads = encode(
        [MessageStart(id="r", model="m"), StreamError(status=529, message="overloaded")]
    )
    error = payloads[-1]
    assert (error["type"], error["message"], error["sequence_number"]) == ("error", "overloaded", 2)
    validate_responses_stream(payloads)


def test_event_before_start_is_an_error():
    with pytest.raises(RuntimeError, match="before message_start"):
        encode([BlockStart(index=0, block=Text(text=""))])
