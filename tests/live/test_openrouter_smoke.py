"""
Live smoke tests: one cheap OpenRouter call per wire format.

    uv run pytest -m live

Needs OPENROUTER_API_KEY in the environment. Each call's cost is appended
to tests/live/spend.jsonl, which is committed after every live run, and
the epic budget guard runs before each call.
"""

import os

import httpx
import pytest

from tests.harness import ledger
from tests.harness.scenarios import TEXT_PROMPT
from tests.harness.sdk import (
    validate_anthropic_message,
    validate_chat_completion,
    validate_response,
)

pytestmark = pytest.mark.live

OPENROUTER_API = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
LIVE_MODEL = "mistralai/mistral-nemo"
ANTHROPIC_VERSION = "2023-06-01"
MAX_TOKENS = 16
TIMEOUT_S = 60.0
LEDGER_SOURCE = "tests/live"

# Fields the official SDK types require but OpenRouter omits, as paths into
# the response body. See "OpenRouter divergences" in tests/fixtures/README.md.
OPENROUTER_RESPONSES_GAPS = [("usage", "input_tokens_details", "cache_write_tokens")]


@pytest.fixture
def openrouter():
    key = os.environ.get(OPENROUTER_KEY_ENV)
    if not key:
        pytest.fail(f"{OPENROUTER_KEY_ENV} must be set to run live tests")
    with httpx.Client(
        base_url=OPENROUTER_API,
        headers={"Authorization": f"Bearer {key}"},
        timeout=TIMEOUT_S,
    ) as client:
        yield client, key


def call_and_record(openrouter, endpoint: str, body: dict, headers=None) -> dict:
    client, key = openrouter
    ledger.check_budget()
    resp = client.post(f"/{endpoint}", json=body, headers=headers or {})
    data = (
        resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    )
    cost = ledger.response_cost(client, key, data)
    ledger.record(
        LEDGER_SOURCE,
        endpoint,
        LIVE_MODEL,
        data.get("id"),
        ledger.UNKNOWN_COST_USD if cost is None else cost,
    )
    assert resp.status_code == 200, f"{endpoint}: HTTP {resp.status_code} {resp.text}"
    return data


def fill_known_gaps(body: dict, gaps) -> dict:
    """Default each known-missing field to 0 so validation still catches anything new."""
    for *parents, leaf in gaps:
        node = body
        for key in parents:
            node = node[key]
        node.setdefault(leaf, 0)
    return body


def test_chat_completions(openrouter):
    body = {
        "model": LIVE_MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": TEXT_PROMPT}],
    }
    completion = validate_chat_completion(call_and_record(openrouter, "chat/completions", body))
    assert completion.choices[0].message.content


def test_messages(openrouter):
    body = {
        "model": LIVE_MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": TEXT_PROMPT}],
    }
    data = call_and_record(openrouter, "messages", body, {"anthropic-version": ANTHROPIC_VERSION})
    message = validate_anthropic_message(data)
    assert any(block.type == "text" and block.text for block in message.content)


def test_responses(openrouter):
    body = {"model": LIVE_MODEL, "max_output_tokens": MAX_TOKENS, "input": TEXT_PROMPT}
    data = call_and_record(openrouter, "responses", body)
    response = validate_response(fill_known_gaps(data, OPENROUTER_RESPONSES_GAPS))
    assert response.output_text
