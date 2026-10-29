"""
The (ingress, backend, scenario) grid, run through the in-process gateway
against replayed backend fixtures. See tests/harness/grid.py.
"""

import json

import pytest

from model_calling.errors import APIStatusError
from model_calling.formats import anthropic
from model_calling.formats import chat_completions as chat
from model_calling.formats import responses
from model_calling.aliases import Target
from model_calling.gateway import client_status, public_message
from tests.harness.grid import (
    GATEWAY_EMBEDDING_INGRESSES,
    GATEWAY_INGRESSES,
    Cell,
    embedding_grid,
    grid,
)
from tests.harness.checks import COMPLETION_CHECKS, MESSAGE_CHECKS
from tests.harness.responses_checks import RESPONSE_CHECKS
from tests.harness.responses_scenarios import REJECTIONS, rejected_name
from tests.harness.gateway import grid_alias
from tests.harness.replay import Exchange, ReplayTransport
from tests.harness.scenarios import ALL_BACKENDS, ALL_SCENARIOS, fixture_for, refused_params
from tests.harness.sdk import (
    SSE_DONE,
    parse_sse,
    validate_anthropic_error,
    validate_anthropic_message,
    validate_anthropic_stream,
    validate_chat_completion,
    validate_chat_completion_chunks,
    validate_embedding_response,
    validate_response,
    validate_responses_stream,
)

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
BAD_REQUEST = 400

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
    "gateway_responses": (
        "/v1/responses",
        lambda scenario, model: scenario.responses_body(model),
        "aresponses",
    ),
    "gateway_embeddings": (
        "/v1/embeddings",
        lambda scenario, model: scenario.body(model),
        "aembedding",
    ),
}
# Fields that differ between two runs of the same exchange. Responses item
# ids derive from the response id, and Ollama invents call ids.
VOLATILE_FIELDS = ("id", "created", "created_at", "item_id", "call_id")


def model(cell: Cell) -> str:
    """The library's name for the cell's model."""
    return ALL_BACKENDS[cell.backend].model_for(ALL_SCENARIOS[cell.scenario])


def request_body(cell: Cell, model_name: str | None = None) -> dict:
    scenario = ALL_SCENARIOS[cell.scenario]
    _, body, _ = INGRESS_CALLS[cell.ingress]
    return body(scenario, model_name or model(cell))


async def post(gateway, cell: Cell):
    """Send the cell to the gateway, naming its model by alias."""
    path = INGRESS_CALLS[cell.ingress][0]
    return await gateway.client.post(path, json=request_body(cell, grid_alias(model(cell))))


async def post_chat_completions(gateway, cell: Cell):
    return await post(gateway, cell)


def assert_no_backend_details(text: str, cell: Cell) -> None:
    """Which backend serves an alias, and as what, is the gateway's business."""
    backend = ALL_BACKENDS[cell.backend]
    prefixed = model(cell)
    leaks = [
        prefixed,
        f"'{prefixed.split('/', 1)[1]}'",
        f"`{prefixed.split('/', 1)[1]}`",
        backend.provider.name,
        backend.gateway_config["base_url"],
    ]
    found = [leak for leak in leaks if leak.lower() in text.lower()]
    assert not found, f"error body names the backend ({found}): {text}"


def assert_backend_auth(cell: Cell, replay: ReplayTransport):
    """A backend with a key gets it as a bearer token; one without gets none."""
    assert replay.requests, "gateway never called the backend"
    key = ALL_BACKENDS[cell.backend].gateway_config.get("api_key")
    expected = f"Bearer {key}" if key else None
    assert replay.requests[0].headers.get("authorization") == expected


async def check_chat_completions(gateway, cell, replay):
    resp = await post_chat_completions(gateway, cell)
    assert resp.status_code == 200, resp.text
    if ALL_SCENARIOS[cell.scenario].stream:
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(resp.text)
        assert events and events[-1] == SSE_DONE, f"stream did not end with [DONE]: {events[-3:]}"
        result = validate_chat_completion_chunks(events[:-1])
    else:
        result = validate_chat_completion(resp.json())
    COMPLETION_CHECKS[cell.scenario](result, grid_alias(model(cell)))


async def check_chat_completions_backend_error(gateway, cell, replay):
    resp = await post_chat_completions(gateway, cell)
    assert (
        resp.status_code >= 400
    ), f"backend error surfaced as HTTP {resp.status_code}: {resp.text}"
    assert_no_backend_details(resp.text, cell)


# --- gateway_messages -------------------------------------------------------


def sse_blocks(text: str) -> list[tuple[str | None, dict]]:
    """Each SSE event as (its `event:` name, its decoded `data`)."""
    blocks = []
    for block in filter(None, text.split("\n\n")):
        fields = dict(line.split(": ", 1) for line in block.split("\n"))
        blocks.append((fields.get("event"), json.loads(fields["data"])))
    return blocks


async def check_messages(gateway, cell, replay):
    resp = await post(gateway, cell)
    assert resp.status_code == 200, resp.text
    if ALL_SCENARIOS[cell.scenario].stream:
        assert resp.headers["content-type"].startswith("text/event-stream")
        blocks = sse_blocks(resp.text)
        assert all(name == data["type"] for name, data in blocks), blocks
        result = validate_anthropic_stream([data for _, data in blocks])
    else:
        result = validate_anthropic_message(resp.json())
    MESSAGE_CHECKS[cell.scenario](result, grid_alias(model(cell)))


async def check_messages_backend_error(gateway, cell, replay):
    resp = await post(gateway, cell)
    recorded = Exchange.load(fixture_for(cell.backend, cell.scenario, cell.form))
    assert resp.status_code == recorded.status, resp.text
    expected = anthropic.error_body(recorded.status, "")["error"]["type"]
    assert validate_anthropic_error(resp.json()).error.type == expected
    assert_no_backend_details(resp.text, cell)


# --- gateway_responses ------------------------------------------------------


async def check_responses(gateway, cell, replay):
    resp = await post(gateway, cell)
    assert resp.status_code == 200, resp.text
    if ALL_SCENARIOS[cell.scenario].stream:
        assert resp.headers["content-type"].startswith("text/event-stream")
        blocks = sse_blocks(resp.text)
        assert all(name == data["type"] for name, data in blocks), blocks
        result = validate_responses_stream([data for _, data in blocks])
    else:
        result = validate_response(resp.json())
    RESPONSE_CHECKS[cell.scenario](result, grid_alias(model(cell)))


async def check_responses_backend_error(gateway, cell, replay):
    resp = await post(gateway, cell)
    recorded = Exchange.load(fixture_for(cell.backend, cell.scenario, cell.form))
    assert resp.status_code == recorded.status, resp.text
    expected = responses.error_body(recorded.status, "")["error"]["type"]
    assert resp.json()["error"]["type"] == expected
    assert_no_backend_details(resp.text, cell)


async def check_responses_rejected(gateway, cell, replay):
    """The stateless ingress refuses what needs state or is out of scope,
    naming it, and never calls the backend."""
    resp = await post(gateway, cell)
    assert resp.status_code == BAD_REQUEST, resp.text
    error = resp.json()["error"]
    assert error["type"] == responses.error_body(BAD_REQUEST, "")["error"]["type"]
    assert rejected_name(cell.scenario) in error["message"]
    replay.check()


# --- gateway_embeddings -----------------------------------------------------


async def check_embeddings(gateway, cell, replay):
    resp = await post(gateway, cell)
    assert resp.status_code == 200, resp.text
    assert validate_embedding_response(resp.json()).model == grid_alias(model(cell))


async def check_embeddings_error(gateway, cell, replay):
    resp = await post(gateway, cell)
    recorded = Exchange.load(fixture_for(cell.backend, cell.scenario, cell.form))
    assert resp.status_code == recorded.status, resp.text
    assert resp.json()["error"]["type"] == chat.error_body(recorded.status, "")["error"]["type"]
    assert_no_backend_details(resp.text, cell)


CHECKS = {
    **{("chat_completions", name): check_chat_completions for name in COMPLETION_CHECKS},
    **{("gateway_messages", name): check_messages for name in MESSAGE_CHECKS},
    ("chat_completions", "backend_error"): check_chat_completions_backend_error,
    ("gateway_messages", "backend_error"): check_messages_backend_error,
    **{("gateway_responses", name): check_responses for name in RESPONSE_CHECKS},
    ("gateway_responses", "backend_error"): check_responses_backend_error,
    **{("gateway_responses", name): check_responses_rejected for name in REJECTIONS},
    ("gateway_embeddings", "embedding"): check_embeddings,
    ("gateway_embeddings", "embedding_error"): check_embeddings_error,
}
GATEWAY_GRID = grid(ingresses=GATEWAY_INGRESSES) + embedding_grid(GATEWAY_EMBEDDING_INGRESSES)
ERROR_TYPES = {
    "chat_completions": chat.error_body,
    "gateway_messages": anthropic.error_body,
    "gateway_responses": responses.error_body,
}


async def check_refused(gateway, cell, replay, monkeypatch):
    """In strict mode, a 400 naming what the backend can't honor, and no
    backend call. (By default the gateway drops such parameters.)"""
    from model_calling import server

    monkeypatch.setattr(server, "DROP_PARAMS", False)
    resp = await post(gateway, cell)
    assert resp.status_code == BAD_REQUEST, resp.text
    error = resp.json()["error"]
    assert error["type"] == ERROR_TYPES[cell.ingress](BAD_REQUEST, "")["error"]["type"]
    params = ", ".join(refused_params(cell.backend, cell.scenario))
    assert error["message"] == f"model {grid_alias(model(cell))!r} cannot honor {params}"
    # Only the model lookups strict mode needed reached the backend.
    replay.check()


@pytest.mark.parametrize("cell", GATEWAY_GRID)
async def test_cell(gateway, cell: Cell, monkeypatch):
    replay = gateway.use(cell.backend, cell.scenario, cell.form)
    if refused_params(cell.backend, cell.scenario):
        await check_refused(gateway, cell, replay, monkeypatch)
        return
    await CHECKS[(cell.ingress, cell.scenario)](gateway, cell, replay)
    if not getattr(ALL_SCENARIOS[cell.scenario], "rejected", False):
        assert_backend_auth(cell, replay)


# --- the gateway answers exactly as the library does ---------------------------


def stable(value):
    """`value` with the fields that change between runs blanked."""
    if isinstance(value, dict):
        return {k: "<volatile>" if k in VOLATILE_FIELDS else stable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


def as_served(value, library_model: str, alias: str):
    """The library's answer with its model name replaced by the alias the
    gateway answers with; nothing else may differ."""
    if isinstance(value, dict):
        return {
            k: alias if k == "model" and v == library_model else as_served(v, library_model, alias)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [as_served(v, library_model, alias) for v in value]
    return value


async def library_answer(library, cell: Cell):
    """What the library returns for the cell, as the gateway would send it:
    a body, a list of stream payloads, or (status, error) for a failure,
    its message in the form the gateway shows clients."""
    library.use(cell.backend, cell.scenario, cell.form)
    method = getattr(library.client, INGRESS_CALLS[cell.ingress][2])
    try:
        result = await method(**request_body(cell))
    except APIStatusError as e:
        target = Target(grid_alias(model(cell)), model(cell), None, None, None)
        return client_status(e), public_message(e.message, target)
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


# Refused cells never produce a body; test_cell checks both refusals.
@pytest.mark.parametrize(
    "cell",
    [
        p
        for p in grid(ingresses=GATEWAY_INGRESSES, known_failures=False)
        if not refused_params(p.values[0].backend, p.values[0].scenario)
    ]
    + embedding_grid(GATEWAY_EMBEDDING_INGRESSES, known_failures=False),
)
async def test_gateway_body_matches_library(gateway, library, cell: Cell):
    gateway.use(cell.backend, cell.scenario, cell.form)
    served = gateway_answer(await post(gateway, cell), cell)
    expected = as_served(await library_answer(library, cell), model(cell), grid_alias(model(cell)))
    assert stable(served) == stable(expected)
