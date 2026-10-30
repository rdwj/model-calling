"""
Live OpenAI Responses through our code: `Client.aresponses` and the
gateway's `/v1/responses` (driven by the `openai` SDK), over
OpenRouter's Responses backend.

    uv run pytest -m live

Needs OPENROUTER_API_KEY in the environment. Each call's cost is appended
to tests/live/spend.jsonl, and the epic budget guard runs before each call.
"""

import httpx2
import openai
import pytest

from model_calling.library import Client
from tests.harness.gateway import GATEWAY_BASE_URL
from tests.harness.responses_checks import response_text, response_tools, response_stream
from tests.harness.scenarios import SCENARIOS
from tests.live.conftest import LIVE_MODEL, SDK_API_KEY, TIMEOUT_S

pytestmark = pytest.mark.live

CHECKS = {"text": response_text, "tools": response_tools}


@pytest.fixture
async def openai_sdk(gateway):
    from model_calling import server

    http = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=server.app), timeout=TIMEOUT_S)
    client = openai.AsyncOpenAI(
        base_url=f"{GATEWAY_BASE_URL}/v1", api_key=SDK_API_KEY, http_client=http
    )
    yield client
    await client.close()
    await http.aclose()


def body(scenario: str, model: str = LIVE_MODEL) -> dict:
    return SCENARIOS[scenario].responses_body(model)


@pytest.mark.parametrize("scenario", CHECKS)
async def test_library_aresponses(spend, scenario):
    async with Client() as client:
        resp = await client.aresponses(**body(scenario))
    spend("library aresponses", resp.id)
    CHECKS[scenario](resp, LIVE_MODEL)


async def test_library_aresponses_stream(spend):
    async with Client() as client:
        events = [e async for e in await client.aresponses(**body("stream"))]
    spend("library aresponses stream", events[0].response.id)
    response_stream(events, LIVE_MODEL)


@pytest.mark.parametrize("scenario", CHECKS)
async def test_gateway_responses(spend, gateway, openai_sdk, scenario):
    alias = gateway.alias(LIVE_MODEL)
    resp = await openai_sdk.responses.create(**body(scenario, alias))
    spend("gateway /v1/responses", resp.id)
    CHECKS[scenario](resp, alias)


async def test_gateway_responses_stream(spend, gateway, openai_sdk):
    alias = gateway.alias(LIVE_MODEL)
    stream = await openai_sdk.responses.create(**body("stream", alias))
    events = [e async for e in stream]
    spend("gateway /v1/responses stream", events[0].response.id)
    response_stream(events, alias)
