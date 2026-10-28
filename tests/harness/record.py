"""
Record backend exchanges into tests/fixtures/ for offline replay.

    uv run python -m tests.harness.record                 # everything missing
    uv run python -m tests.harness.record --backend ollama --scenario tools --force

Sends each scenario's native request straight to the backend (not through
our code, so fixtures capture the backend's real behavior even where ours
is broken). Ollama must be running with qwen3:4b and nomic-embed-text
pulled. OpenRouter needs
OPENROUTER_API_KEY in the environment; each paid call is appended to
tests/live/spend.jsonl and the epic budget guard runs first. vLLM (both its Chat
Completions and its Anthropic Messages endpoint) is read from
VLLM_BASE_URL and VLLM_MODEL (VLLM_API_KEY optional); fixtures store
the replay URL and model from scenarios.py instead, so no deployment
details reach the repo. VLLM_VERIFY_TLS=false skips certificate checks for
vLLM only, for an endpoint reached through a port-forward. Request headers
are never written, account ids are scrubbed from response bodies, and a
fixture containing a key is refused.
"""

import argparse
import datetime
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

import httpx

from tests.harness import ledger
from tests.harness.replay import Exchange, fixture_path
from tests.harness.scenarios import (
    ALL_BACKENDS,
    ALL_SCENARIOS,
    EMBEDDING_SCENARIOS,
    VLLM_URL,
    Backend,
    EmbeddingBackend,
    EmbeddingScenario,
    Scenario,
    recorded_scenarios,
)


def _scenarios_for(backend) -> dict:
    if isinstance(backend, EmbeddingBackend):
        return EMBEDDING_SCENARIOS
    return {s.name: s for s in recorded_scenarios(backend)}


OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
VLLM_BASE_URL_ENV = "VLLM_BASE_URL"
VLLM_MODEL_ENV = "VLLM_MODEL"
VLLM_KEY_ENV = "VLLM_API_KEY"
VLLM_VERIFY_TLS_ENV = "VLLM_VERIFY_TLS"
REQUEST_TIMEOUT = 300.0
LEDGER_SOURCE = "record_fixtures"
# Account identifiers OpenRouter puts in response bodies. Not secrets, but
# the repo is public.
SCRUBBED_RESPONSE_KEYS = ("user_id",)
SCRUBBED = "scrubbed"


def _openrouter_key() -> str:
    return _require_env(OPENROUTER_KEY_ENV, "record OpenRouter fixtures")


@dataclass(frozen=True)
class Target:
    """Where a recording really goes, when that differs from the replay URL."""

    url: str
    model: Optional[str] = None
    api_key: Optional[str] = None
    verify_tls: bool = True


def _require_env(name: str, why: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"{name} is not set; it is required to {why}.")
    return value


def _target(backend: Backend) -> Target:
    if not backend.endpoint.startswith(VLLM_URL):
        return Target(url=backend.endpoint)
    why = "record vLLM fixtures"
    base_url = _require_env(VLLM_BASE_URL_ENV, why).rstrip("/")
    verify = os.environ.get(VLLM_VERIFY_TLS_ENV, "true").strip().lower()
    if verify not in ("true", "false"):
        sys.exit(f"{VLLM_VERIFY_TLS_ENV} must be 'true' or 'false', not {verify!r}.")
    return Target(
        url=f"{base_url}/v1{backend.endpoint[len(VLLM_URL):]}",
        model=_require_env(VLLM_MODEL_ENV, why),
        api_key=os.environ.get(VLLM_KEY_ENV) or None,
        verify_tls=verify == "true",
    )


def _replace_model(value, real: str, stored: str):
    """Swap the deployment's model id for the stored one, wherever it appears."""
    if isinstance(value, dict):
        return {k: _replace_model(v, real, stored) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_model(v, real, stored) for v in value]
    if isinstance(value, str) and value.startswith("data:"):
        payload = value[len("data:") :].strip()
        if payload.startswith("{"):
            return "data: " + json.dumps(_replace_model(json.loads(payload), real, stored))
    return stored if value == real else value


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


def record_one(
    client: httpx.Client,
    backend: Backend | EmbeddingBackend,
    scenario: Scenario | EmbeddingScenario,
    target: Target,
) -> Exchange:
    body = backend.native_request(scenario)
    sent = body
    if target.model is not None and not scenario.missing_model:
        sent = {**body, "model": target.model}
    headers = {}
    key = None
    if backend.needs_openrouter_key:
        key = _openrouter_key()
        ledger.check_budget()
        headers["Authorization"] = f"Bearer {key}"
    elif target.api_key:
        headers["Authorization"] = f"Bearer {target.api_key}"

    with client.stream("POST", target.url, json=sent, headers=headers) as resp:
        content_type = resp.headers.get("content-type", "").split(";")[0]
        if scenario.stream and resp.status_code == 200:
            lines, response_json = list(resp.iter_lines()), None
        else:
            resp.read()
            lines, response_json = None, resp.json()

    if key is not None:
        if lines is not None:
            response_id, cost = _stream_cost(lines)
            if cost is None:
                cost = ledger.response_cost(client, key, {"id": response_id})
        elif resp.status_code == 200:
            response_id = response_json.get("id")
            cost = ledger.response_cost(client, key, response_json)
        else:
            # OpenRouter doesn't bill rejected requests.
            response_id, cost = None, 0.0
        ledger.record(
            LEDGER_SOURCE,
            backend.endpoint,
            body["model"],
            response_id,
            ledger.UNKNOWN_COST_USD if cost is None else cost,
        )

    if isinstance(response_json, dict):
        for k in SCRUBBED_RESPONSE_KEYS:
            if k in response_json:
                response_json[k] = SCRUBBED
    if target.model is not None:
        stored = backend.native_model_for(scenario)
        response_json = _replace_model(response_json, target.model, stored)
        lines = _replace_model(lines, target.model, stored)

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
    saved = json.dumps(exchange.to_dict())
    for secret in (key, target.api_key):
        if secret and secret in saved:
            raise RuntimeError(
                f"Refusing to save {backend.name}/{scenario.name}: it contains a key"
            )
    return exchange


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", choices=sorted(ALL_BACKENDS), action="append")
    parser.add_argument("--scenario", choices=sorted(ALL_SCENARIOS), action="append")
    parser.add_argument("--force", action="store_true", help="re-record existing fixtures")
    args = parser.parse_args(argv)

    backends = [ALL_BACKENDS[n] for n in (args.backend or sorted(ALL_BACKENDS))]
    spent_before = ledger.total_spend()

    for backend in backends:
        family = _scenarios_for(backend)
        names = [n for n in (args.scenario or sorted(family)) if n in family]
        todo = []
        for scenario in (family[n] for n in names):
            path = fixture_path(backend.name, scenario.name)
            if path.exists() and not args.force:
                print(f"skip   {path} (exists; --force to re-record)")
            else:
                todo.append((scenario, path))
        if not todo:
            continue
        target = _target(backend)
        with httpx.Client(timeout=REQUEST_TIMEOUT, verify=target.verify_tls) as client:
            for scenario, path in todo:
                exchange = record_one(client, backend, scenario, target)
                exchange.save(path)
                print(f"saved  {path} (HTTP {exchange.status})")

    print(f"OpenRouter spend this run: ${ledger.total_spend() - spent_before:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
