"""
Self-tests for the offline harness: replay, SSE parsing, ledger, fixtures.
"""

import json
import re

import httpx
import pytest

from tests.harness import ledger
from tests.harness.grid import Cell
from tests.harness.replay import FIXTURES_DIR, Exchange, ReplayTransport
from tests.harness.scenarios import (
    BACKENDS,
    EMBEDDING_BACKENDS,
    EMBEDDING_SCENARIOS,
    GRID_BACKENDS,
    SEED,
    recorded_scenarios,
)
from tests.harness.sdk import (
    SSE_DONE,
    parse_sse,
    validate_anthropic_request,
    validate_chat_completion_request,
)

URL = "https://backend.test/v1/chat"


def exchange(status=200, body=None, url=URL, sent=None):
    return Exchange("POST", url, sent, status, "application/json", response_json=body or {})


async def test_replay_serves_matching_exchanges_in_order():
    replay = ReplayTransport([exchange(body={"n": 1}, sent={"a": 1}), exchange(body={"n": 2})])
    async with httpx.AsyncClient(transport=replay) as client:
        first = await client.post(URL, json={"a": 1})
        second = await client.post(URL + "?ignored=query")
    assert [first.json()["n"], second.json()["n"]] == [1, 2]
    assert replay.sent_json(0) == {"a": 1}


def test_replay_picks_the_exchange_whose_body_matches():
    replay = ReplayTransport(
        [exchange(body={"n": 1}, sent={"a": 1}), exchange(body={"n": 2}, sent={"a": 2})]
    )
    with httpx.Client(transport=replay) as client:
        assert client.post(URL, json={"a": 2}).json() == {"n": 2}


@pytest.mark.parametrize(
    "tamper, match",
    [
        (lambda body: body.update(max_tokens=1), r"max_tokens: recorded 1, sent 256"),
        (lambda body: body.pop("stream"), r"stream: recorded '<absent>', sent False"),
        (lambda body: body["messages"].append({"role": "user", "content": "x"}), r"messages: "),
    ],
    ids=["changed", "removed", "nested"],
)
async def test_replay_rejects_a_tampered_fixture(gateway, tamper, match):
    """The grid fails when a fixture's recorded request differs from what we send.

    The gateway swallows the transport's error into an HTTP error response,
    so the failure has to surface through `check()`.
    """
    from tests.test_gateway_grid import post_chat_completions

    replay = gateway.use("vllm", "text")
    tamper(replay._pending[0].request_json)
    await post_chat_completions(gateway, Cell("chat_completions", "vllm", "text"))
    with pytest.raises(AssertionError, match=f"differs from the recording.*{match}"):
        replay.check()
    replay.failures.clear()
    replay._pending.clear()


def test_replay_rejects_unrecorded_requests():
    replay = ReplayTransport([exchange()])
    with httpx.Client(transport=replay) as client:
        with pytest.raises(AssertionError, match="No recorded exchange for POST https://other"):
            client.post("https://other.test/v1/chat")


def test_streaming_exchange_replays_lines():
    lines = ['data: {"x": 1}', "", "data: [DONE]", ""]
    ex = Exchange("POST", URL, {}, 200, "text/event-stream", lines=lines)
    assert parse_sse(ex.to_response().text) == [{"x": 1}, SSE_DONE]


@pytest.mark.parametrize(
    "text, expected",
    [
        ('data: {"a": 1}\n\ndata: [DONE]\n\n', [{"a": 1}, SSE_DONE]),
        ('event: message_start\ndata: {"type": "message_start"}\n\n', [{"type": "message_start"}]),
        ('data: {"a": 1}\r\n\r\n: comment\n\n', [{"a": 1}]),
        ("data: [DONE]  \n\n", [SSE_DONE]),
        ("data: \n\n", []),
    ],
)
def test_parse_sse(text, expected):
    assert parse_sse(text) == expected


def test_parse_sse_rejects_non_json_data():
    with pytest.raises(ValueError, match="not JSON"):
        parse_sse("data: {broken\n\n")


@pytest.mark.parametrize(
    "costs, budget, allowed",
    [([], 5.0, True), ([1.0, 3.99], 5.0, True), ([2.5, 2.5], 5.0, False), ([6.0], 5.0, False)],
)
def test_ledger_budget_guard(tmp_path, costs, budget, allowed):
    path = tmp_path / "spend.jsonl"
    for cost in costs:
        ledger.record("test", "endpoint", "model", "id", cost, path=path)
    assert ledger.total_spend(path) == pytest.approx(sum(costs))
    if allowed:
        assert ledger.check_budget(budget, path) == pytest.approx(budget - sum(costs))
    else:
        with pytest.raises(ledger.BudgetExceeded):
            ledger.check_budget(budget, path)


def test_ledger_reports_malformed_lines(tmp_path):
    path = tmp_path / "spend.jsonl"
    path.write_text('{"cost_usd": 0.1}\nnot json\n')
    with pytest.raises(ValueError, match=r"spend.jsonl:2"):
        ledger.total_spend(path)


def test_committed_ledger_is_valid_and_under_budget():
    assert ledger.total_spend() < ledger.EPIC_BUDGET_USD


ALL_FIXTURES = sorted(FIXTURES_DIR.glob("*/*.json"))
SECRET_PATTERNS = [
    r"sk-or-",
    r"(?i)\bbearer\s+\S",
    r"(?i)\"authorization\"",
    r"user_[0-9A-Za-z]{20,}",
]


def test_every_grid_cell_has_a_fixture():
    expected = {
        FIXTURES_DIR / name / f"{s.name}.json"
        for name, b in BACKENDS.items()
        for s in recorded_scenarios(b)
    }
    expected |= {
        FIXTURES_DIR / b / f"{s}.json" for b in EMBEDDING_BACKENDS for s in EMBEDDING_SCENARIOS
    }
    assert expected <= set(ALL_FIXTURES), sorted(map(str, expected - set(ALL_FIXTURES)))


def test_refused_cells_have_no_fixture():
    """A refused cell never reaches the backend, so a fixture for it is stale."""
    stale = [
        FIXTURES_DIR / name / f"{scenario}.json"
        for name, b in BACKENDS.items()
        for scenario in b.refuses
    ]
    assert not [p for p in stale if p.exists()], stale


OPENAI_COMPAT_BACKENDS = ("openai_compat", "vllm")
# (scenario, backends, path into the recorded native body, expected value):
# the Chat Completions-only fields reach each backend that can take them.
NATIVE_FIELDS = [
    ("reasoning_effort", OPENAI_COMPAT_BACKENDS, ("reasoning_effort",), "low"),
    ("reasoning_effort", ("ollama",), ("think",), True),
    ("tool_required", OPENAI_COMPAT_BACKENDS, ("tool_choice",), "required"),
    (
        "tool_named",
        OPENAI_COMPAT_BACKENDS,
        ("tool_choice",),
        {"type": "function", "function": {"name": "get_weather"}},
    ),
    ("parallel_tools_off", OPENAI_COMPAT_BACKENDS, ("parallel_tool_calls",), False),
    ("tool_strict", OPENAI_COMPAT_BACKENDS, ("tools", 0, "function", "strict"), True),
    (
        "tool_parameterless",
        GRID_BACKENDS,
        ("tools", 1, "function", "parameters"),
        {"type": "object", "properties": {}},
    ),
    ("max_completion_tokens", GRID_BACKENDS, ("max_tokens",), 256),
    ("max_completion_tokens", GRID_BACKENDS, ("seed",), SEED),
]
# Ollama takes sampling settings under `options`.
OLLAMA_OPTION_FIELDS = {"seed": "seed", "max_tokens": "num_predict"}


def _dig(body, path):
    for key in path:
        body = body[key]
    return body


@pytest.mark.parametrize(
    "backend, scenario, path, expected",
    [
        (backend, scenario, path, expected)
        for scenario, backends, path, expected in NATIVE_FIELDS
        for backend in backends
    ],
)
def test_native_body_carries_field(backend, scenario, path, expected):
    sent = Exchange.load(FIXTURES_DIR / backend / f"{scenario}.json").request_json
    if backend == "ollama" and path[0] in OLLAMA_OPTION_FIELDS:
        path = ("options", OLLAMA_OPTION_FIELDS[path[0]])
    assert _dig(sent, path) == expected, sent


@pytest.mark.parametrize("path", ALL_FIXTURES, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_no_native_body_uses_legacy_function_calling(path):
    sent = Exchange.load(path).request_json or {}
    assert not {"functions", "function_call"} & set(sent), sent


@pytest.mark.parametrize("path", ALL_FIXTURES, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_fixture_is_scrubbed_and_loads(path):
    text = path.read_text(encoding="utf-8")
    for pattern in SECRET_PATTERNS:
        assert not re.search(pattern, text), f"{path} matches secret pattern {pattern!r}"
    ex = Exchange.load(path)
    assert (ex.response_json is None) != (ex.lines is None)
    json.dumps(ex.to_dict())


@pytest.mark.parametrize(
    "message, match",
    [
        ({"role": "user", "content": "hi", "typo": 1}, "unknown or coerced fields"),
        ({"role": "user", "content": [{"type": "tool_result"}]}, "tool_use_id"),
    ],
)
def test_validate_anthropic_request_is_strict(message, match):
    body = {"model": "m", "max_tokens": 1, "messages": [message]}
    with pytest.raises(ValueError, match=match):
        validate_anthropic_request(body)


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"typo": 1}, "unknown or coerced fields"),
        ({"tool_choice": {"type": "function"}}, "function"),
    ],
)
def test_validate_chat_completion_request_is_strict(extra, match):
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], **extra}
    with pytest.raises(ValueError, match=match):
        validate_chat_completion_request(body)


def test_replay_check_reports_unused_exchanges():
    replay = ReplayTransport([exchange()])
    with pytest.raises(AssertionError, match="never requested"):
        replay.check()


def test_replay_compares_types_exactly():
    replay = ReplayTransport([exchange(sent={"stream": True})])
    with httpx.Client(transport=replay) as client:
        with pytest.raises(AssertionError, match="stream: recorded True, sent 1"):
            client.post(URL, json={"stream": 1})


def test_replay_records_a_non_json_body():
    replay = ReplayTransport([exchange(sent={})])
    with httpx.Client(transport=replay) as client:
        with pytest.raises(AssertionError, match="isn't JSON"):
            client.post(URL, content=b"not json")
    assert replay.failures
