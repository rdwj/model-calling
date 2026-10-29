"""
The gateway driven by the official SDK clients, the way apps call it:
`AsyncAnthropic(base_url=...)` on the Messages ingress, streaming through
the SDK's own `messages.stream` helper, and `AsyncOpenAI(base_url=...)` on
the Chat Completions ingress, streaming through the SDK's own iterator. See
tests/harness/grid.py.
"""

import anthropic
import httpx2
import openai
import pytest
from openai.lib.streaming.chat import ChatCompletionStreamState

from model_calling.formats import anthropic as anthropic_format
from model_calling.formats import chat_completions as chat_format
from tests.harness.checks import (
    COMPLETION_CHECKS,
    FINAL_COMPLETION_CHECKS,
    FINAL_MESSAGE_CHECKS,
    MESSAGE_CHECKS,
)
from tests.harness.gateway import GATEWAY_BASE_URL, grid_alias
from tests.harness.grid import SDK_INGRESSES, Cell, grid
from tests.harness.replay import Exchange
from tests.harness.scenarios import BACKENDS, SCENARIOS, fixture_for, refused_params
from tests.harness.sdk import validate_anthropic_request, validate_chat_completion_request

# The gateway takes no client key; the SDKs require one.
SDK_API_KEY = "sdk-grid-not-a-real-key"
# The openai SDK's base URL includes the API version.
OPENAI_BASE_URL = f"{GATEWAY_BASE_URL}/v1"
BAD_REQUEST = 400
# Stream events the SDK's helper adds to the raw wire events.
RAW_EVENT_TYPES = {
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
}


@pytest.fixture
async def http(gateway):
    """An httpx2 client over the ASGI app the `gateway` fixture serves. Both
    SDKs run on httpx2, their fork of httpx."""
    from model_calling import server

    client = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=server.app))
    yield client
    await client.aclose()


@pytest.fixture
async def sdk(http):
    client = anthropic.AsyncAnthropic(
        base_url=GATEWAY_BASE_URL, api_key=SDK_API_KEY, http_client=http
    )
    yield client
    await client.close()


@pytest.fixture
async def openai_sdk(http):
    client = openai.AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=SDK_API_KEY, http_client=http)
    yield client
    await client.close()


def model(cell: Cell) -> str:
    """The alias the gateway serves the cell's model under (see `Gateway.use`)."""
    return grid_alias(BACKENDS[cell.backend].model_for(SCENARIOS[cell.scenario]))


def recorded_status(cell: Cell) -> int:
    return Exchange.load(fixture_for(cell.backend, cell.scenario, cell.form)).status


# --- sdk_messages -----------------------------------------------------------


def request_body(cell: Cell) -> dict:
    body = SCENARIOS[cell.scenario].messages_body(model(cell))
    # The scenario is a request the SDK's own types accept.
    validate_anthropic_request(body)
    return body


async def check_stream(sdk, cell: Cell):
    body = {k: v for k, v in request_body(cell).items() if k != "stream"}
    async with sdk.messages.stream(**body) as stream:
        events = [e async for e in stream if e.type in RAW_EVENT_TYPES]
        final = await stream.get_final_message()
    MESSAGE_CHECKS[cell.scenario](events, model(cell))
    # The SDK assembles the same message from our deltas.
    FINAL_MESSAGE_CHECKS[cell.scenario](final, model(cell))
    assert final.usage.output_tokens == events[-2].usage.output_tokens


async def check_backend_error(sdk, cell: Cell):
    status = recorded_status(cell)
    with pytest.raises(anthropic.APIStatusError) as err:
        await sdk.messages.create(**request_body(cell))
    assert err.value.status_code == status
    expected = anthropic_format.error_body(status, "")["error"]["type"]
    assert err.value.body["error"]["type"] == expected


async def check_messages(sdk, cell: Cell):
    if cell.scenario == "backend_error":
        await check_backend_error(sdk, cell)
    elif SCENARIOS[cell.scenario].stream:
        await check_stream(sdk, cell)
    else:
        msg = await sdk.messages.create(**request_body(cell))
        MESSAGE_CHECKS[cell.scenario](msg, model(cell))


# Fields of a Messages request that count_tokens doesn't take.
NOT_COUNTED = ("max_tokens", "stream", "metadata", "stop_sequences")


@pytest.mark.parametrize("scenario", ["system_blocks", "tool_error", "image", "thinking"])
async def test_count_tokens(gateway, sdk, scenario):
    alias = gateway.alias(BACKENDS["ollama"].model_for(SCENARIOS[scenario]))
    body = SCENARIOS[scenario].messages_body(alias)
    counted = await sdk.messages.count_tokens(
        **{k: v for k, v in body.items() if k not in NOT_COUNTED}
    )
    assert counted.input_tokens > 0


# --- sdk_completion ---------------------------------------------------------


def completion_body(cell: Cell) -> dict:
    body = SCENARIOS[cell.scenario].chat_completions_body(model(cell))
    validate_chat_completion_request(body)
    return body


async def check_completion_stream(openai_sdk, cell: Cell):
    body = completion_body(cell)
    stream = await openai_sdk.chat.completions.create(**body)
    state = ChatCompletionStreamState(input_tools=body.get("tools", openai.omit))
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
        state.handle_chunk(chunk)
    COMPLETION_CHECKS[cell.scenario](chunks, model(cell))
    # The SDK assembles the same completion from our chunks.
    FINAL_COMPLETION_CHECKS[cell.scenario](state.get_final_completion(), model(cell))


async def check_completion_backend_error(openai_sdk, cell: Cell):
    status = recorded_status(cell)
    with pytest.raises(openai.APIStatusError) as err:
        await openai_sdk.chat.completions.create(**completion_body(cell))
    assert err.value.status_code == status
    assert err.value.type == chat_format.error_body(status, "")["error"]["type"]


async def check_completion(openai_sdk, cell: Cell):
    if cell.scenario == "backend_error":
        await check_completion_backend_error(openai_sdk, cell)
    elif SCENARIOS[cell.scenario].stream:
        await check_completion_stream(openai_sdk, cell)
    else:
        resp = await openai_sdk.chat.completions.create(**completion_body(cell))
        COMPLETION_CHECKS[cell.scenario](resp, model(cell))


# --- both -------------------------------------------------------------------

# Ingress -> (the SDK's create call for a cell, the SDK's 400 error).
REFUSAL_CALLS = {
    "sdk_messages": (
        lambda sdk, openai_sdk, cell: sdk.messages.create(**request_body(cell)),
        anthropic.BadRequestError,
    ),
    "sdk_completion": (
        lambda sdk, openai_sdk, cell: openai_sdk.chat.completions.create(**completion_body(cell)),
        openai.BadRequestError,
    ),
}


async def check_refused(sdk, openai_sdk, cell: Cell, replay, monkeypatch):
    """In strict mode, the SDK gets a 400 naming what the backend can't
    honor, and the backend is never called."""
    from model_calling import server

    monkeypatch.setattr(server, "DROP_PARAMS", False)
    call, error = REFUSAL_CALLS[cell.ingress]
    with pytest.raises(error) as err:
        await call(sdk, openai_sdk, cell)
    assert err.value.status_code == BAD_REQUEST
    params = ", ".join(refused_params(cell.backend, cell.scenario))
    assert f"{BACKENDS[cell.backend].provider.name} cannot honor {params}" in err.value.message
    assert not replay.requests, "a refused request reached the backend"


@pytest.mark.parametrize("cell", grid(ingresses=SDK_INGRESSES))
async def test_cell(gateway, sdk, openai_sdk, cell: Cell, monkeypatch):
    replay = gateway.use(cell.backend, cell.scenario, cell.form)
    if refused_params(cell.backend, cell.scenario):
        await check_refused(sdk, openai_sdk, cell, replay, monkeypatch)
    elif cell.ingress == "sdk_messages":
        await check_messages(sdk, cell)
    else:
        await check_completion(openai_sdk, cell)
