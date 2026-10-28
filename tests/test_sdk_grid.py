"""
The gateway's Messages ingress driven by the official `anthropic` client,
the way Claude Code-style apps call it: `AsyncAnthropic(base_url=...)`
pointed at the in-process gateway, streaming through the SDK's own
`messages.stream` helper. See tests/harness/grid.py.
"""

import anthropic
import httpx2
import pytest

from model_calling.formats import anthropic as anthropic_format
from tests.harness.checks import FINAL_MESSAGE_CHECKS, MESSAGE_CHECKS
from tests.harness.gateway import GATEWAY_BASE_URL
from tests.harness.grid import SDK_INGRESSES, Cell, grid
from tests.harness.replay import Exchange
from tests.harness.scenarios import BACKENDS, SCENARIOS, fixture_for
from tests.harness.sdk import validate_anthropic_request

# The gateway takes no client key; the SDK requires one.
SDK_API_KEY = "sdk-grid-not-a-real-key"
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
async def sdk(gateway):
    """An `AsyncAnthropic` client whose requests reach the in-process gateway.

    anthropic 2.x runs on httpx2, its own fork of httpx, so it gets an httpx2
    client over the same ASGI app the `gateway` fixture serves.
    """
    from model_calling import server

    http = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=server.app))
    client = anthropic.AsyncAnthropic(
        base_url=GATEWAY_BASE_URL, api_key=SDK_API_KEY, http_client=http
    )
    yield client
    await client.close()
    await http.aclose()


def model(cell: Cell) -> str:
    return BACKENDS[cell.backend].model_for(SCENARIOS[cell.scenario])


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
    recorded = Exchange.load(fixture_for(cell.backend, cell.scenario))
    with pytest.raises(anthropic.APIStatusError) as err:
        await sdk.messages.create(**request_body(cell))
    assert err.value.status_code == recorded.status
    expected = anthropic_format.error_body(recorded.status, "")["error"]["type"]
    assert err.value.body["error"]["type"] == expected


# Fields of a Messages request that count_tokens doesn't take.
NOT_COUNTED = ("max_tokens", "stream", "metadata", "stop_sequences")


@pytest.mark.parametrize("scenario", ["system_blocks", "tool_error", "image", "thinking"])
async def test_count_tokens(sdk, scenario):
    body = SCENARIOS[scenario].messages_body(BACKENDS["ollama"].model_for(SCENARIOS[scenario]))
    counted = await sdk.messages.count_tokens(
        **{k: v for k, v in body.items() if k not in NOT_COUNTED}
    )
    assert counted.input_tokens > 0


@pytest.mark.parametrize("cell", grid(ingresses=SDK_INGRESSES))
async def test_cell(gateway, sdk, cell: Cell):
    gateway.use(cell.backend, cell.scenario)
    if cell.scenario == "backend_error":
        await check_backend_error(sdk, cell)
    elif SCENARIOS[cell.scenario].stream:
        await check_stream(sdk, cell)
    else:
        msg = await sdk.messages.create(**request_body(cell))
        MESSAGE_CHECKS[cell.scenario](msg, model(cell))
