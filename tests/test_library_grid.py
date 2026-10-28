"""
The library API's cells of the (ingress, backend, scenario) grid: `Client`
calls against replayed backend fixtures. See tests/harness/grid.py.
"""

import pytest

from model_calling.errors import APIStatusError
from tests.harness.grid import LIBRARY_INGRESSES, Cell, grid
from tests.harness.replay import Exchange, fixture_path
from tests.harness.scenarios import BACKENDS, SCENARIOS
from tests.harness.sdk import (
    validate_anthropic_message,
    validate_anthropic_stream,
    validate_chat_completion,
    validate_chat_completion_chunks,
)

EXPECTED_TEXT = "pong"
EXPECTED_TOOL = ("get_weather", {"location": "Paris"})


def model(cell: Cell) -> str:
    return BACKENDS[cell.backend].model_for(SCENARIOS[cell.scenario])


async def completion(library, cell: Cell):
    body = SCENARIOS[cell.scenario].chat_completions_body(model(cell))
    return await library.client.acompletion(**body)


async def messages(library, cell: Cell):
    body = SCENARIOS[cell.scenario].messages_body(model(cell))
    return await library.client.amessages(**body)


async def check_backend_error(call, library, cell):
    recorded = Exchange.load(fixture_path(cell.backend, cell.scenario))
    with pytest.raises(APIStatusError) as err:
        await call(library, cell)
    assert err.value.status == recorded.status
    assert err.value.body == recorded.response_json


# --- library_completion ---------------------------------------------------


async def check_completion_text(library, cell):
    resp = await completion(library, cell)
    validate_chat_completion(resp.to_dict())
    choice = resp.choices[0]
    assert EXPECTED_TEXT in (choice.message.content or "").lower()
    assert choice.finish_reason == "stop"
    assert resp.model == model(cell)
    assert resp.usage is not None and resp.usage.total_tokens > 0


async def check_completion_tools(library, cell):
    import json

    resp = await completion(library, cell)
    validate_chat_completion(resp.to_dict())
    choice = resp.choices[0]
    assert choice.message.tool_calls, f"no tool_calls in {choice.message}"
    call = choice.message.tool_calls[0]
    assert (call.function.name, json.loads(call.function.arguments)) == EXPECTED_TOOL
    assert choice.finish_reason == "tool_calls"


async def check_completion_stream(library, cell):
    chunks = [c async for c in await completion(library, cell)]
    validate_chat_completion_chunks([c.to_dict() for c in chunks])
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert EXPECTED_TEXT in text.lower()
    finish = [
        c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason
    ]
    assert finish == ["stop"], finish
    assert len({c.id for c in chunks}) == 1, "chunks of one stream must share an id"
    assert {c.model for c in chunks} == {model(cell)}


async def check_completion_backend_error(library, cell):
    await check_backend_error(completion, library, cell)


# --- library_messages -----------------------------------------------------


async def check_messages_text(library, cell):
    msg = await messages(library, cell)
    validate_anthropic_message(msg.to_dict())
    text = "".join(b.text for b in msg.content if b.type == "text")
    assert EXPECTED_TEXT in text.lower()
    assert msg.stop_reason == "end_turn"
    assert msg.model == model(cell)
    assert msg.usage.output_tokens > 0


async def check_messages_tools(library, cell):
    msg = await messages(library, cell)
    validate_anthropic_message(msg.to_dict())
    call = next((b for b in msg.content if b.type == "tool_use"), None)
    assert call is not None, f"no tool_use in {msg.content}"
    assert (call.name, call.input) == EXPECTED_TOOL
    assert msg.stop_reason == "tool_use"


async def check_messages_stream(library, cell):
    events = [e async for e in await messages(library, cell)]
    validate_anthropic_stream([e.to_dict() for e in events])
    kinds = [e.type for e in events]
    assert kinds[0] == "message_start" and kinds[-2:] == ["message_delta", "message_stop"], kinds
    assert events[0].message.model == model(cell)
    open_blocks = set()
    for e in events:
        if e.type == "content_block_start":
            assert e.index not in open_blocks
            open_blocks.add(e.index)
        elif e.type in ("content_block_delta", "content_block_stop"):
            assert e.index in open_blocks, f"{e.type} for unopened block {e.index}"
            if e.type == "content_block_stop":
                open_blocks.remove(e.index)
    assert not open_blocks, f"blocks never stopped: {open_blocks}"
    text = "".join(
        e.delta.text
        for e in events
        if e.type == "content_block_delta" and e.delta.type == "text_delta"
    )
    assert EXPECTED_TEXT in text.lower()
    assert events[-2].delta.stop_reason == "end_turn"


async def check_messages_backend_error(library, cell):
    await check_backend_error(messages, library, cell)


CHECKS = {
    ("library_completion", "text"): check_completion_text,
    ("library_completion", "tools"): check_completion_tools,
    ("library_completion", "stream"): check_completion_stream,
    ("library_completion", "backend_error"): check_completion_backend_error,
    ("library_messages", "text"): check_messages_text,
    ("library_messages", "tools"): check_messages_tools,
    ("library_messages", "stream"): check_messages_stream,
    ("library_messages", "backend_error"): check_messages_backend_error,
}


@pytest.mark.parametrize("cell", grid(ingresses=LIBRARY_INGRESSES))
async def test_cell(library, cell: Cell):
    replay = library.use(cell.backend, cell.scenario)
    await CHECKS[(cell.ingress, cell.scenario)](library, cell)
    assert replay.requests, "client never called the backend"
    key = BACKENDS[cell.backend].gateway_config.get("api_key")
    assert replay.requests[0].headers.get("authorization") == (f"Bearer {key}" if key else None)
