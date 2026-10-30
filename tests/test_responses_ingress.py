"""
The OpenAI Responses ingress beyond the grid: instructions as the system
prompt, a stream that fails partway, the sync API, and a previous
response's output sent back as input.
"""

import json

import httpx
import pytest

from model_calling.errors import APIError, InternalServerError
from model_calling.formats import responses
from model_calling.library import Client
from model_calling.types.responses import Response
from tests.harness.replay import Exchange, ReplayTransport, fixture_path
from tests.harness.scenarios import BACKENDS, CHAT_COMPLETIONS, MESSAGES, SCENARIOS, Scenario
from tests.harness.sdk import parse_sse, validate_responses_stream

OLLAMA = BACKENDS["ollama"]
INSTRUCTIONS = "You are a terse assistant."
PROMPT = SCENARIOS["text"].prompt
# The same request with a system prompt, in each form.
SYSTEM_PROMPT = Scenario(
    "instructions",
    PROMPT,
    chat_messages=(
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": PROMPT},
    ),
    messages_extra={"system": INSTRUCTIONS},
)


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_instructions_reach_the_backend_as_its_system_prompt(backend):
    """`instructions` sends each backend the body a Chat Completions system
    message does, or for a Messages backend a Messages `system` string."""
    target = BACKENDS[backend]
    model = target.native_model_for(SYSTEM_PROMPT)
    provider = target.native_provider(SYSTEM_PROMPT)
    body = {"model": model, "input": PROMPT, "instructions": INSTRUCTIONS, "max_output_tokens": 256}
    sent = provider.build_request(responses.request_to_internal({**body, "stream": False}))
    form = MESSAGES if target.speaks == MESSAGES else CHAT_COMPLETIONS
    assert sent == provider.build_request(SYSTEM_PROMPT.native_internal(model, form))


def ollama_stream(*lines: dict) -> httpx.MockTransport:
    text = "".join(json.dumps(line) + "\n" for line in lines)
    return httpx.MockTransport(lambda req: httpx.Response(200, text=text))


TEXT_LINE = {"model": "m", "message": {"role": "assistant", "content": "po"}}
CRASH_LINE = {"error": "model crashed"}


async def test_gateway_stream_failing_partway_ends_with_a_numbered_error_event(gateway):
    gateway.backends.config["ollama"].update(transport=ollama_stream(TEXT_LINE, CRASH_LINE))
    body = {"model": gateway.alias("ollama/m"), "input": "hi", "stream": True}
    resp = await gateway.client.post("/v1/responses", json=body)
    assert resp.status_code == 200, resp.text
    events = parse_sse(resp.text)
    validate_responses_stream(events)
    assert [e["sequence_number"] for e in events] == list(range(len(events)))
    error = events[-1]
    assert error["type"] == "error" and "model crashed" in error["message"]
    assert "ollama" not in error["message"].lower()


async def test_library_stream_failing_partway_raises():
    transport = ollama_stream(TEXT_LINE, {"error": "server overloaded"})
    async with Client(config={"ollama": {"transport": transport}}) as client:
        events = await client.aresponses("ollama/m", "hi", stream=True)
        with pytest.raises(APIError, match="overloaded"):
            async for _ in events:
                pass


def replay(*scenarios: str) -> ReplayTransport:
    return ReplayTransport([Exchange.load(fixture_path("ollama", s)) for s in scenarios])


def ollama_client(transport) -> Client:
    return Client(config={"ollama": {**OLLAMA.gateway_config, "transport": transport}})


def body(scenario: str) -> dict:
    return SCENARIOS[scenario].responses_body(OLLAMA.model_for(SCENARIOS[scenario]))


def test_sync_responses_and_stream():
    transport = replay("text", "stream")
    with ollama_client(transport) as client:
        resp = client.responses(**body("text"))
        assert isinstance(resp, Response)
        assert "pong" in resp.output_text.lower()
        events = client.responses(**body("stream"))
        assert not hasattr(events, "__aiter__")
        text = "".join(e.delta for e in events if e.type == "response.output_text.delta")
    assert "pong" in text.lower()
    transport.check()


async def test_previous_output_goes_back_as_input():
    """Stateless multi-turn: the client sends the last response's output
    items back, and they reach the backend as the assistant's turn."""
    async with ollama_client(replay("tools")) as client:
        first = await client.aresponses(**body("tools"))
    call = next(item for item in first.output if item.type == "function_call")
    sent = []

    def capture(request):
        sent.append(json.loads(request.content))
        return httpx.Response(500, json={"error": "stop here"})

    turns = [
        {"role": "user", "content": SCENARIOS["tools"].prompt},
        *first.output,
        {"type": "function_call_output", "call_id": call.call_id, "output": "sunny"},
    ]
    async with ollama_client(httpx.MockTransport(capture)) as client:
        with pytest.raises(InternalServerError):
            await client.aresponses(OLLAMA.model_for(SCENARIOS["tools"]), turns)
    assistant, result = sent[0]["messages"][1:]
    assert assistant["tool_calls"][0]["function"]["name"] == call.name
    assert result == {"role": "tool", "content": "sunny", "tool_name": call.name}
