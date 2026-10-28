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
from tests.harness.scenarios import BACKENDS, SCENARIOS
from tests.harness.sdk import SSE_DONE, parse_sse, validate_anthropic_request

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
    expected = {FIXTURES_DIR / b / f"{s}.json" for b in BACKENDS for s in SCENARIOS}
    assert expected <= set(ALL_FIXTURES), sorted(map(str, expected - set(ALL_FIXTURES)))


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
