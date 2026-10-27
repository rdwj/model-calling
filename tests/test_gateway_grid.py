"""
The (ingress, backend, scenario) grid, run through the in-process gateway
against replayed backend fixtures. See tests/harness/grid.py.
"""

import json

import pytest

from tests.harness.grid import Cell, grid
from tests.harness.replay import ReplayTransport
from tests.harness.scenarios import BACKENDS, REPLAY_API_KEY, SCENARIOS
from tests.harness.sdk import (
    SSE_DONE,
    parse_sse,
    validate_chat_completion,
    validate_chat_completion_chunks,
)

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
EXPECTED_TEXT = "pong"
EXPECTED_TOOL = ("get_weather", {"location": "Paris"})


async def post_chat_completions(gateway, cell: Cell):
    backend = BACKENDS[cell.backend]
    body = SCENARIOS[cell.scenario].chat_completions_body(backend.gateway_model)
    return await gateway.client.post(CHAT_COMPLETIONS_PATH, json=body)


def assert_backend_auth(cell: Cell, replay: ReplayTransport):
    """Hosted backends must receive the configured credential."""
    if cell.backend != "openai_compat":
        return
    assert replay.requests, "gateway never called the backend"
    assert replay.requests[0].headers.get("authorization") == f"Bearer {REPLAY_API_KEY}"


async def check_chat_completions_text(gateway, cell, replay):
    resp = await post_chat_completions(gateway, cell)
    assert resp.status_code == 200, resp.text
    completion = validate_chat_completion(resp.json())
    choice = completion.choices[0]
    assert EXPECTED_TEXT in (choice.message.content or "").lower()
    assert choice.finish_reason == "stop"
    assert completion.model == BACKENDS[cell.backend].gateway_model
    assert completion.usage is not None and completion.usage.total_tokens > 0


async def check_chat_completions_tools(gateway, cell, replay):
    resp = await post_chat_completions(gateway, cell)
    assert resp.status_code == 200, resp.text
    completion = validate_chat_completion(resp.json())
    choice = completion.choices[0]
    assert choice.message.function_call is None, "legacy function_call emitted"
    assert choice.message.tool_calls, f"no tool_calls in {choice.message}"
    call = choice.message.tool_calls[0]
    assert (call.function.name, json.loads(call.function.arguments)) == EXPECTED_TOOL
    assert choice.finish_reason == "tool_calls"


async def check_chat_completions_stream(gateway, cell, replay):
    resp = await post_chat_completions(gateway, cell)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(resp.text)
    assert events and events[-1] == SSE_DONE, f"stream did not end with [DONE]: {events[-3:]}"
    chunks = validate_chat_completion_chunks(events[:-1])
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert EXPECTED_TEXT in text.lower()
    finish = [
        c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason
    ]
    # OpenRouter repeats finish_reason on its usage chunk, so allow repeats.
    assert finish and set(finish) == {"stop"}, finish
    assert len({c.id for c in chunks}) == 1, "chunks of one stream must share an id"
    assert {c.model for c in chunks} == {BACKENDS[cell.backend].gateway_model}


async def check_chat_completions_backend_error(gateway, cell, replay):
    resp = await post_chat_completions(gateway, cell)
    assert (
        resp.status_code >= 400
    ), f"backend error surfaced as HTTP {resp.status_code}: {resp.text}"


CHECKS = {
    ("chat_completions", "text"): check_chat_completions_text,
    ("chat_completions", "tools"): check_chat_completions_tools,
    ("chat_completions", "stream"): check_chat_completions_stream,
    ("chat_completions", "backend_error"): check_chat_completions_backend_error,
}


@pytest.mark.parametrize("cell", grid())
async def test_cell(gateway, cell: Cell):
    replay = gateway.use(cell.backend, cell.scenario)
    await CHECKS[(cell.ingress, cell.scenario)](gateway, cell, replay)
    assert_backend_auth(cell, replay)
