"""
Live OpenAI Chat Completions through our code: `Client.acompletion` and the
gateway's `/v1/chat/completions` (driven by the `openai` SDK), over
OpenRouter's Chat Completions backend.

    uv run pytest -m live

Needs OPENROUTER_API_KEY in the environment. Each call's cost is appended
to tests/live/spend.jsonl, and the epic budget guard runs before each call.
"""

import httpx2
import openai
import pytest

from model_calling.library import Client
from tests.harness.checks import completion_stream_usage, completion_text, completion_tools
from tests.harness.gateway import GATEWAY_BASE_URL
from tests.harness.scenarios import SCENARIOS
from tests.live.conftest import LIVE_MODEL, SDK_API_KEY, TIMEOUT_S

pytestmark = pytest.mark.live

CHECKS = {"text": completion_text, "tools": completion_tools}


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
    return SCENARIOS[scenario].chat_completions_body(model)


@pytest.mark.parametrize("scenario", CHECKS)
async def test_library_acompletion(spend, scenario):
    async with Client() as client:
        resp = await client.acompletion(**body(scenario))
    spend("library acompletion", resp.id)
    CHECKS[scenario](resp, LIVE_MODEL)


async def test_library_acompletion_stream_usage(spend):
    async with Client() as client:
        chunks = [c async for c in await client.acompletion(**body("stream_usage"))]
    spend("library acompletion stream", chunks[0].id)
    completion_stream_usage(chunks, LIVE_MODEL)


@pytest.mark.parametrize("scenario", CHECKS)
async def test_gateway_chat_completions(spend, gateway, openai_sdk, scenario):
    alias = gateway.alias(LIVE_MODEL)
    resp = await openai_sdk.chat.completions.create(**body(scenario, alias))
    spend("gateway /v1/chat/completions", resp.id)
    CHECKS[scenario](resp, alias)


async def test_gateway_chat_completions_stream_usage(spend, gateway, openai_sdk):
    alias = gateway.alias(LIVE_MODEL)
    stream = await openai_sdk.chat.completions.create(**body("stream_usage", alias))
    chunks = [c async for c in stream]
    spend("gateway /v1/chat/completions stream", chunks[0].id)
    completion_stream_usage(chunks, alias)
