"""
Shared fixtures for the live tests, which call OpenRouter through our code.
"""

import os

import httpx
import pytest

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

        def record(endpoint: str, response_id: str) -> None:
            cost = ledger.response_cost(client, key, {"id": response_id})
            ledger.record(
                LEDGER_SOURCE,
                endpoint,
                LIVE_MODEL,
                response_id,
                ledger.UNKNOWN_COST_USD if cost is None else cost,
            )

        yield record
