"""
Record backend exchanges into tests/fixtures/ for offline replay.

    uv run python -m tests.harness.record                 # everything missing
    uv run python -m tests.harness.record --backend ollama --scenario tools --force

Sends each scenario's native request straight to the backend (not through
our code, so fixtures capture the backend's real behavior even where ours
is broken). Ollama must be running with qwen3:4b pulled. OpenRouter needs
OPENROUTER_API_KEY in the environment; each paid call is appended to
tests/live/spend.jsonl and the epic budget guard runs first. Request
headers are never written, account ids are scrubbed from response bodies,
and a fixture containing the key is refused.
"""

import argparse
import datetime
import json
import os
import sys
from typing import Optional

import httpx

from tests.harness import ledger
from tests.harness.replay import Exchange, fixture_path
from tests.harness.scenarios import BACKENDS, SCENARIOS, Backend, Scenario

OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
REQUEST_TIMEOUT = 300.0
LEDGER_SOURCE = "record_fixtures"
# Account identifiers OpenRouter puts in response bodies. Not secrets, but
# the repo is public.
SCRUBBED_RESPONSE_KEYS = ("user_id",)
SCRUBBED = "scrubbed"


def _openrouter_key() -> str:
    key = os.environ.get(OPENROUTER_KEY_ENV)
    if not key:
        sys.exit(f"{OPENROUTER_KEY_ENV} is not set; it is required to record OpenRouter fixtures.")
    return key


def _stream_cost(lines) -> tuple[Optional[str], Optional[float]]:
    """Find the generation id and usage.cost in an SSE chat stream."""
    response_id, cost = None, None
    for line in lines:
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        chunk = json.loads(line[len("data:") :])
        response_id = chunk.get("id", response_id)
        usage = chunk.get("usage") or {}
        if "cost" in usage:
            cost = usage["cost"]
    return response_id, cost


def record_one(client: httpx.Client, backend: Backend, scenario: Scenario) -> Exchange:
    body = backend.native_request(scenario)
    headers = {}
    key = None
    if backend.needs_openrouter_key:
        key = _openrouter_key()
        ledger.check_budget()
        headers["Authorization"] = f"Bearer {key}"

    with client.stream("POST", backend.endpoint, json=body, headers=headers) as resp:
        content_type = resp.headers.get("content-type", "").split(";")[0]
        if scenario.stream and resp.status_code == 200:
            lines, response_json = list(resp.iter_lines()), None
        else:
            resp.read()
            lines, response_json = None, resp.json()

    if key is not None:
        if lines is not None:
            response_id, cost = _stream_cost(lines)
        else:
            response_id = response_json.get("id")
            cost = (response_json.get("usage") or {}).get("cost")
        if cost is None and response_id and resp.status_code == 200:
            cost = ledger.lookup_generation_cost(client, key, response_id)
        ledger.record(
            LEDGER_SOURCE, backend.endpoint, body["model"], response_id, float(cost or 0.0)
        )

    if isinstance(response_json, dict):
        for k in SCRUBBED_RESPONSE_KEYS:
            if k in response_json:
                response_json[k] = SCRUBBED

    exchange = Exchange(
        method="POST",
        url=backend.endpoint,
        request_json=body,
        status=resp.status_code,
        content_type=content_type,
        response_json=response_json,
        lines=lines,
        meta={
            "backend": backend.name,
            "scenario": scenario.name,
            "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "recorder": "tests/harness/record.py",
        },
    )
    if key is not None and key in json.dumps(exchange.to_dict()):
        raise RuntimeError(f"Refusing to save {backend.name}/{scenario.name}: it contains the key")
    return exchange


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", choices=sorted(BACKENDS), action="append")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), action="append")
    parser.add_argument("--force", action="store_true", help="re-record existing fixtures")
    args = parser.parse_args(argv)

    backends = [BACKENDS[n] for n in (args.backend or sorted(BACKENDS))]
    scenarios = [SCENARIOS[n] for n in (args.scenario or sorted(SCENARIOS))]
    spent_before = ledger.total_spend()

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        for backend in backends:
            for scenario in scenarios:
                path = fixture_path(backend.name, scenario.name)
                if path.exists() and not args.force:
                    print(f"skip   {path} (exists; --force to re-record)")
                    continue
                exchange = record_one(client, backend, scenario)
                exchange.save(path)
                print(f"saved  {path} (HTTP {exchange.status})")

    print(f"OpenRouter spend this run: ${ledger.total_spend() - spent_before:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
