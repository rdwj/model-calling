"""`POST /v1/messages/count_tokens` and the estimate behind it."""

import pytest

from model_calling.formats.anthropic import request_to_internal
from model_calling.token_estimate import (
    CHARS_PER_TOKEN,
    TOKENS_PER_BLOCK,
    TOKENS_PER_IMAGE,
    TOKENS_PER_MESSAGE,
    estimate_input_tokens,
)
from tests.harness.scenarios import ANTHROPIC_WEATHER_TOOL, IMAGE_B64, SCENARIOS
from tests.harness.sdk import validate_anthropic_error
from anthropic.types import MessageTokensCount

COUNT_TOKENS_PATH = "/v1/messages/count_tokens"
MODEL = "ollama/qwen3:4b"
BASE = {"model": MODEL, "messages": [{"role": "user", "content": "Hi"}]}
IMAGE_BLOCK = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": IMAGE_B64},
}


def estimate(body: dict) -> int:
    return estimate_input_tokens(request_to_internal(body))


def test_estimate_of_one_short_turn():
    text = "x" * (CHARS_PER_TOKEN * 5)
    body = {"model": MODEL, "messages": [{"role": "user", "content": text}]}
    assert estimate(body) == TOKENS_PER_MESSAGE + TOKENS_PER_BLOCK + 5


def test_empty_request_estimates_zero():
    assert estimate({"model": MODEL, "messages": []}) == 0


def test_tool_result_without_content():
    turn = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1"}]}
    body = {"model": MODEL, "messages": [turn]}
    assert estimate(body) == TOKENS_PER_MESSAGE + TOKENS_PER_BLOCK


# Each addition to BASE, and the least it must add to the estimate.
ADDITIONS = {
    "system": ({"system": "You are a helpful assistant."}, 1),
    "system_blocks": ({"system": [{"type": "text", "text": "Be terse."}] * 2}, 2),
    "tools": ({"tools": [ANTHROPIC_WEATHER_TOOL]}, 10),
    "server_tool": ({"tools": [{"type": "web_search_20250305", "name": "web_search"}]}, 10),
    "longer_text": ({"messages": [{"role": "user", "content": "word " * 200}]}, 200),
    "image": (
        {"messages": [{"role": "user", "content": [IMAGE_BLOCK, {"type": "text", "text": "Hi"}]}]},
        TOKENS_PER_IMAGE,
    ),
    "tool_history": (
        {"messages": list(SCENARIOS["tool_error"].anthropic_messages)},
        TOKENS_PER_MESSAGE * 2,
    ),
    "thinking_history": (
        {
            "messages": [
                *BASE["messages"],
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "a" * 400, "signature": ""},
                        {"type": "text", "text": "Hello."},
                    ],
                },
                {"role": "user", "content": "Again."},
            ]
        },
        100,
    ),
}


@pytest.mark.parametrize("extra, at_least", ADDITIONS.values(), ids=ADDITIONS.keys())
def test_estimate_counts(extra, at_least):
    assert estimate({**BASE, **extra}) - estimate(BASE) >= at_least


@pytest.fixture
def served(gateway):
    """A request body naming MODEL by the alias the gateway serves it under."""
    return {**BASE, "model": gateway.alias(MODEL)}


@pytest.mark.parametrize("extra, at_least", ADDITIONS.values(), ids=ADDITIONS.keys())
async def test_endpoint_answers_with_the_estimate(gateway, served, extra, at_least):
    body = {**served, **extra}
    resp = await gateway.client.post(COUNT_TOKENS_PATH, json=body)
    assert resp.status_code == 200, resp.text
    assert MessageTokensCount.model_validate(resp.json()).input_tokens == estimate(body)


async def test_endpoint_calls_no_backend(gateway, served):
    # Replays nothing: the gateway fixture fails at teardown on any backend request.
    replay = gateway.use("ollama", "text")
    resp = await gateway.client.post(COUNT_TOKENS_PATH, json=served)
    assert resp.status_code == 200
    assert replay.requests == []
    gateway.replays.remove(replay)


BAD_BODIES = {
    "not_an_object": (lambda served: [served], "must be a JSON object"),
    "no_model": (lambda served: {"messages": BASE["messages"]}, "model is required"),
    "no_messages": (lambda served: {"model": served["model"]}, "invalid Anthropic request"),
    "messages_not_a_list": (
        lambda served: {**served, "messages": "hi"},
        "invalid Anthropic request",
    ),
    "system_image": (lambda served: {**served, "system": [IMAGE_BLOCK]}, "only text blocks"),
}


@pytest.mark.parametrize("make_body, message", BAD_BODIES.values(), ids=BAD_BODIES.keys())
async def test_endpoint_rejects(gateway, served, make_body, message):
    resp = await gateway.client.post(COUNT_TOKENS_PATH, json=make_body(served))
    assert resp.status_code == 400, resp.text
    error = validate_anthropic_error(resp.json()).error
    assert error.type == "invalid_request_error"
    assert message in error.message


@pytest.mark.parametrize("name", [MODEL, "cohere/command-a", "no-such-alias"])
async def test_endpoint_refuses_a_name_that_is_not_an_alias(gateway, served, name):
    resp = await gateway.client.post(COUNT_TOKENS_PATH, json={**served, "model": name})
    assert resp.status_code == 404, resp.text
    error = validate_anthropic_error(resp.json()).error
    assert error.type == "not_found_error"
    assert repr(name) in error.message
