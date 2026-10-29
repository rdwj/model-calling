"""
Shared fixtures for the live tests, which call OpenRouter through our code.
"""

import os

import httpx
import pytest

from model_calling.callbacks import Callback
from model_calling.core import MessageDelta
from model_calling.formats.anthropic import FORMAT as ANTHROPIC_FORMAT
from tests.harness import ledger

OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
LIVE_MODEL = "openrouter/mistralai/mistral-nemo"
LEDGER_SOURCE = "tests/live"
TIMEOUT_S = 60.0
# The gateway takes no client key; the SDKs require one.
SDK_API_KEY = "live-not-a-real-key"


@pytest.fixture
def spend():
    """Record a call's cost from OpenRouter's generation lookup, by response id."""
    key = os.environ.get(OPENROUTER_KEY_ENV)
    if not key:
        pytest.fail(f"{OPENROUTER_KEY_ENV} must be set to run live tests")
    ledger.check_budget()
    with httpx.Client(timeout=TIMEOUT_S) as client:

        def record(
            endpoint: str, response_id: str, model: str = LIVE_MODEL, cost: float | None = None
        ) -> None:
            """`cost` when the response reported it (OpenRouter's Messages
            endpoint does, and its generation lookup rejects `msg_` ids)."""
            if cost is None:
                cost = ledger.response_cost(client, key, {"id": response_id})
            ledger.record(
                LEDGER_SOURCE,
                endpoint,
                model,
                response_id,
                ledger.UNKNOWN_COST_USD if cost is None else cost,
            )

        yield record


class CostCapture(Callback):
    """The `usage.cost` OpenRouter reports, from the native body of a call
    or the usage on a stream's message delta."""

    def __init__(self):
        self.cost: float | None = None

    async def on_request_end(self, model, messages, response, **kwargs):
        if isinstance(response, dict):
            self.cost = (response.get("usage") or {}).get("cost", self.cost)

    async def on_stream_chunk(self, model, messages, chunk, **kwargs):
        if isinstance(chunk, MessageDelta) and chunk.usage is not None:
            self.cost = chunk.usage.extras_for(ANTHROPIC_FORMAT).get("cost", self.cost)


@pytest.fixture
def cost():
    return CostCapture()
