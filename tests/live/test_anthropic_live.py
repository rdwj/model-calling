"""
Live Anthropic backend: a Claude model on OpenRouter's `/api/v1/messages`,
reached through the `anthropic/` prefix by `Client.amessages` and by the
gateway's `/v1/messages` (driven by the `anthropic` SDK).

    uv run pytest -m live

Needs OPENROUTER_API_KEY in the environment. Claude costs far more than the
other live models, so this runs one call per scenario and path. Each call's
cost, which OpenRouter reports in the response, is appended to
tests/live/spend.jsonl.
"""

import os

import anthropic
import httpx2
import pytest

from model_calling.library import Client
from tests.harness.checks import MESSAGE_CHECKS
from tests.harness.gateway import GATEWAY_BASE_URL
from tests.harness.scenarios import OPENROUTER_URL, SCENARIOS
from tests.live.conftest import OPENROUTER_KEY_ENV, SDK_API_KEY, TIMEOUT_S

pytestmark = pytest.mark.live

# The cheapest Claude on OpenRouter (2026-10-28: $0.10/M in, $0.50/M out).
MODEL = "anthropic/anthropic/claude-haiku-5.5"
# Haiku 5.5 there never thinks, whatever the thinking config (checked
# 2026-10-28); Haiku 4.5 does ($1/M in, $5/M out).
THINKING_MODEL = "anthropic/anthropic/claude-haiku-4.5"
SCENARIO_NAMES = ("text", "tools", "stream_tools", "thinking")


def model(scenario: str) -> str:
    return THINKING_MODEL if scenario == "thinking" else MODEL


def config() -> dict:
    return {"base_url": OPENROUTER_URL, "api_key": os.environ.get(OPENROUTER_KEY_ENV, "")}


def body(scenario: str, name: str | None = None) -> dict:
    return SCENARIOS[scenario].messages_body(name or model(scenario))


@pytest.fixture
async def sdk(gateway, cost):
    from model_calling import server

    gateway.backends.config["anthropic"].update(config())
    gateway.backends.callbacks.append(cost)
    http = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=server.app), timeout=TIMEOUT_S)
    client = anthropic.AsyncAnthropic(
        base_url=GATEWAY_BASE_URL, api_key=SDK_API_KEY, http_client=http
    )
    yield client
    await client.close()
    await http.aclose()


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
async def test_library_amessages(spend, cost, scenario):
    async with Client(config={"anthropic": config()}, callbacks=[cost]) as client:
        result = await client.amessages(**body(scenario))
        if SCENARIOS[scenario].stream:
            result = [event async for event in result]
    response_id = result[0].message.id if isinstance(result, list) else result.id
    spend("library amessages (anthropic)", response_id, model(scenario), cost.cost)
    MESSAGE_CHECKS[scenario](result, model(scenario))


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
async def test_gateway_messages(spend, cost, gateway, sdk, scenario):
    alias = gateway.alias(model(scenario))
    result = await sdk.messages.create(**body(scenario, alias))
    if SCENARIOS[scenario].stream:
        result = [event async for event in result]
    response_id = result[0].message.id if isinstance(result, list) else result.id
    spend("gateway /v1/messages (anthropic)", response_id, model(scenario), cost.cost)
    MESSAGE_CHECKS[scenario](result, alias)
