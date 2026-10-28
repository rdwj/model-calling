"""
Live Anthropic Messages through our code: `Client.amessages` and the
gateway's `/v1/messages` (driven by the `anthropic` SDK), both over
OpenRouter's Chat Completions backend, so a real model sees our Messages →
Chat Completions translation.

    uv run pytest -m live

Needs OPENROUTER_API_KEY in the environment. Each call's cost is appended
to tests/live/spend.jsonl, and the epic budget guard runs before each call.
"""

import anthropic
import httpx2
import pytest

from model_calling.library import Client
from tests.harness.checks import message_text, message_tools
from tests.harness.gateway import GATEWAY_BASE_URL
from tests.harness.scenarios import SCENARIOS
from tests.live.conftest import LIVE_MODEL, SDK_API_KEY, TIMEOUT_S

pytestmark = pytest.mark.live

CHECKS = {"text": message_text, "tools": message_tools}


@pytest.fixture
async def sdk(gateway):
    from model_calling import server

    http = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=server.app), timeout=TIMEOUT_S)
    client = anthropic.AsyncAnthropic(
        base_url=GATEWAY_BASE_URL, api_key=SDK_API_KEY, http_client=http
    )
    yield client
    await client.close()
    await http.aclose()


@pytest.mark.parametrize("scenario", CHECKS)
async def test_library_amessages(spend, scenario):
    async with Client() as client:
        msg = await client.amessages(**SCENARIOS[scenario].messages_body(LIVE_MODEL))
    spend("library amessages", msg.id)
    CHECKS[scenario](msg, LIVE_MODEL)


@pytest.mark.parametrize("scenario", CHECKS)
async def test_gateway_messages(spend, sdk, scenario):
    msg = await sdk.messages.create(**SCENARIOS[scenario].messages_body(LIVE_MODEL))
    spend("gateway /v1/messages", msg.id)
    CHECKS[scenario](msg, LIVE_MODEL)


async def test_gateway_messages_stream_tools(spend, sdk):
    body = SCENARIOS["tools"].messages_body(LIVE_MODEL)
    body.pop("stream")
    async with sdk.messages.stream(**body) as stream:
        final = await stream.get_final_message()
    spend("gateway /v1/messages stream", final.id)
    message_tools(final, LIVE_MODEL)
