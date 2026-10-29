"""
The vllm_messages fixtures validate against the anthropic SDK types.

The grid replays them through the Anthropic provider; these tests check
the recordings themselves. Each place vLLM departs from the SDK types is
listed in tests/fixtures/README.md.
"""

import pytest

from tests.harness.replay import Exchange, fixture_path
from tests.harness.sdk import (
    parse_sse,
    validate_anthropic_error,
    validate_anthropic_message,
    validate_anthropic_stream,
)

BACKEND = "vllm_messages"


def load(scenario: str) -> Exchange:
    return Exchange.load(fixture_path(BACKEND, scenario))


def test_text():
    message = validate_anthropic_message(load("text").response_json)
    assert [b.type for b in message.content] == ["thinking", "text"]
    assert message.content[0].signature, "vLLM signs its thinking blocks"
    assert message.content[1].text == "pong"
    assert message.stop_reason == "end_turn"


def test_tools():
    message = validate_anthropic_message(load("tools").response_json)
    call = next(b for b in message.content if b.type == "tool_use")
    assert (call.name, call.input) == ("get_weather", {"location": "Paris"})
    assert message.stop_reason == "tool_use"


def test_stream():
    ex = load("stream")
    events = parse_sse(ex.body_bytes().decode())
    start = next(e for e in events if e["type"] == "content_block_start")
    # vLLM omits the thinking block's empty signature here (see the README).
    assert start["content_block"] == {"type": "thinking", "thinking": ""}
    start["content_block"]["signature"] = ""
    typed = validate_anthropic_stream(events)

    kinds = [e.type for e in typed]
    assert kinds[0] == "message_start" and kinds[-2:] == ["message_delta", "message_stop"]
    deltas = [e.delta for e in typed if e.type == "content_block_delta"]
    assert {d.type for d in deltas} == {"thinking_delta", "signature_delta", "text_delta"}
    assert "".join(d.text for d in deltas if d.type == "text_delta") == "pong"
    assert typed[-2].delta.stop_reason == "end_turn"


@pytest.mark.xfail(
    strict=True,
    reason="vLLM's error type is 'NotFoundError', not 'not_found_error' (fixtures README)",
)
def test_backend_error():
    ex = load("backend_error")
    assert ex.status == 404
    validate_anthropic_error(ex.response_json)
