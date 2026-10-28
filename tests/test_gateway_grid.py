"""
The (ingress, backend, scenario) grid, run through the in-process gateway
against replayed backend fixtures. See tests/harness/grid.py.
"""

import json

import pytest

from model_calling.errors import APIStatusError
from model_calling.formats import anthropic
from model_calling.formats import chat_completions as chat
from model_calling.gateway import client_status
from tests.harness.grid import (
    GATEWAY_EMBEDDING_INGRESSES,
    GATEWAY_INGRESSES,
    Cell,
    embedding_grid,
    grid,
)
from tests.harness.replay import Exchange, ReplayTransport, fixture_path
from tests.harness.scenarios import ALL_BACKENDS, ALL_SCENARIOS, BACKENDS
from tests.harness.sdk import (
    SSE_DONE,
    parse_sse,
    validate_anthropic_error,
    validate_anthropic_message,
    validate_anthropic_stream,
    validate_chat_completion,
    validate_chat_completion_chunks,
    validate_embedding_response,
)

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
EXPECTED_TEXT = "pong"
EXPECTED_TOOL = ("get_weather", {"location": "Paris"})

# Ingress -> (gateway path, request body for a scenario and model, library method).
INGRESS_CALLS = {
    "chat_completions": (
        CHAT_COMPLETIONS_PATH,
        lambda scenario, model: scenario.chat_completions_body(model),
        "acompletion",
    ),
    "gateway_messages": (
        "/v1/messages",
        lambda scenario, model: scenario.messages_body(model),
        "amessages",
    ),
    "gateway_embeddings": (
        "/v1/embeddings",
        lambda scenario, model: scenario.body(model),
        "aembedding",
    ),
}
# Fields that differ between two runs of the same exchange.
VOLATILE_FIELDS = ("id", "created")


def request_body(cell: Cell) -> dict:
    scenario = ALL_SCENARIOS[cell.scenario]
    _, body, _ = INGRESS_CALLS[cell.ingress]
    return body(scenario, ALL_BACKENDS[cell.backend].model_for(scenario))


async def post(gateway, cell: Cell):
    path = INGRESS_CALLS[cell.ingress][0]
    return await gateway.client.post(path, json=request_body(cell))


async def post_chat_completions(gateway, cell: Cell):
    return await post(gateway, cell)


def assert_backend_auth(cell: Cell, replay: ReplayTransport):
    """A backend with a key gets it as a bearer token; one without gets none."""
    assert replay.requests, "gateway never called the backend"
    key = ALL_BACKENDS[cell.backend].gateway_config.get("api_key")
    expected = f"Bearer {key}" if key else None
    assert replay.requests[0].headers.get("authorization") == expected


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


# --- gateway_messages -------------------------------------------------------


def sse_blocks(text: str) -> list[tuple[str | None, dict]]:
    """Each SSE event as (its `event:` name, its decoded `data`)."""
    blocks = []
    for block in filter(None, text.split("\n\n")):
        fields = dict(line.split(": ", 1) for line in block.split("\n"))
        blocks.append((fields.get("event"), json.loads(fields["data"])))
    return blocks


async def check_messages_body(gateway, cell, replay):
    resp = await post(gateway, cell)
    assert resp.status_code == 200, resp.text
    validate_anthropic_message(resp.json())


async def check_messages_stream(gateway, cell, replay):
    resp = await post(gateway, cell)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    blocks = sse_blocks(resp.text)
    assert all(name == data["type"] for name, data in blocks), blocks
    events = [data for _, data in blocks]
    validate_anthropic_stream(events)
    assert [e["type"] for e in events[-2:]] == ["message_delta", "message_stop"]


async def check_messages_backend_error(gateway, cell, replay):
    resp = await post(gateway, cell)
    recorded = Exchange.load(fixture_path(cell.backend, cell.scenario))
    assert resp.status_code == recorded.status, resp.text
    expected = anthropic.error_body(recorded.status, "")["error"]["type"]
    assert validate_anthropic_error(resp.json()).error.type == expected


# --- gateway_embeddings -----------------------------------------------------


async def check_embeddings(gateway, cell, replay):
    resp = await post(gateway, cell)
    assert resp.status_code == 200, resp.text
    validate_embedding_response(resp.json())


async def check_embeddings_error(gateway, cell, replay):
    resp = await post(gateway, cell)
    recorded = Exchange.load(fixture_path(cell.backend, cell.scenario))
    assert resp.status_code == recorded.status, resp.text
    assert resp.json()["error"]["type"] == chat.error_body(recorded.status, "")["error"]["type"]


CHECKS = {
    ("chat_completions", "text"): check_chat_completions_text,
    ("chat_completions", "tools"): check_chat_completions_tools,
    ("chat_completions", "stream"): check_chat_completions_stream,
    ("chat_completions", "backend_error"): check_chat_completions_backend_error,
    ("gateway_messages", "text"): check_messages_body,
    ("gateway_messages", "tools"): check_messages_body,
    ("gateway_messages", "stream"): check_messages_stream,
    ("gateway_messages", "backend_error"): check_messages_backend_error,
    ("gateway_embeddings", "embedding"): check_embeddings,
    ("gateway_embeddings", "embedding_error"): check_embeddings_error,
}
GATEWAY_GRID = grid(ingresses=GATEWAY_INGRESSES) + embedding_grid(GATEWAY_EMBEDDING_INGRESSES)


@pytest.mark.parametrize("cell", GATEWAY_GRID)
async def test_cell(gateway, cell: Cell):
    replay = gateway.use(cell.backend, cell.scenario)
    await CHECKS[(cell.ingress, cell.scenario)](gateway, cell, replay)
    assert_backend_auth(cell, replay)


# --- the gateway answers exactly as the library does ---------------------------


def stable(value):
    """`value` with the fields that change between runs blanked."""
    if isinstance(value, dict):
        return {k: "<volatile>" if k in VOLATILE_FIELDS else stable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


async def library_answer(library, cell: Cell):
    """What the library returns for the cell, as the gateway would send it:
    a body, a list of stream payloads, or (status, error) for a failure."""
    library.use(cell.backend, cell.scenario)
    method = getattr(library.client, INGRESS_CALLS[cell.ingress][2])
    try:
        result = await method(**request_body(cell))
    except APIStatusError as e:
        return client_status(e), e.message
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return [item.to_dict() async for item in result]


def gateway_answer(resp, cell: Cell):
    if resp.status_code >= 400:
        body = resp.json()
        return resp.status_code, body["error"]["message"]
    if not resp.headers["content-type"].startswith("text/event-stream"):
        return resp.json()
    return [e for e in parse_sse(resp.text) if e != SSE_DONE]


@pytest.mark.parametrize("cell", GATEWAY_GRID)
async def test_gateway_body_matches_library(gateway, library, cell: Cell):
    gateway.use(cell.backend, cell.scenario)
    served = gateway_answer(await post(gateway, cell), cell)
    assert stable(served) == stable(await library_answer(library, cell))
