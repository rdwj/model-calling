"""
Parametrize tests over (ingress, backend, scenario) cells.

Later phases add ingresses and backends here and delete entries from
KNOWN_FAILURES as they fix them. Known failures are strict xfails: once a
cell starts passing, the run fails until its entry is removed.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import pytest

from tests.harness.scenarios import BACKENDS, SCENARIOS

INGRESSES = ("chat_completions",)

# (ingress, backend, scenario) -> reason, naming the phase that fixes it.
_HOSTED_CALLBACK_CRASH = (
    "Phase 2: hosted adapters pass **translated_request to callbacks, so the body's "
    "'model' key collides with the positional model name and every call returns 500"
)
KNOWN_FAILURES: Dict[Tuple[str, str, str], str] = {
    ("chat_completions", "ollama", "tools"): (
        "Phase 4: Ollama tool calls keep finish_reason 'stop' instead of 'tool_calls'"
    ),
    ("chat_completions", "ollama", "stream"): (
        "Phase 2: server.stream_response bypasses the adapter and never sends [DONE] "
        "for NDJSON backends"
    ),
    ("chat_completions", "ollama", "backend_error"): (
        "Phase 2: backend errors return HTTP 200 with the backend's error body"
    ),
    ("chat_completions", "openai_compat", "text"): _HOSTED_CALLBACK_CRASH,
    ("chat_completions", "openai_compat", "tools"): _HOSTED_CALLBACK_CRASH,
    ("chat_completions", "openai_compat", "backend_error"): _HOSTED_CALLBACK_CRASH,
    ("chat_completions", "openai_compat", "stream"): (
        "Phase 2: server.stream_response appends /v1/chat/completions to the hosted "
        "adapter's full endpoint URL and sends no auth header"
    ),
}


@dataclass(frozen=True)
class Cell:
    ingress: str
    backend: str
    scenario: str

    @property
    def id(self) -> str:
        return f"{self.ingress}-{self.backend}-{self.scenario}"


def grid(
    scenarios: Iterable[str] = tuple(SCENARIOS),
    ingresses: Iterable[str] = INGRESSES,
    backends: Iterable[str] = tuple(BACKENDS),
) -> list:
    params = []
    for ingress in ingresses:
        for backend in backends:
            for scenario in scenarios:
                cell = Cell(ingress, backend, scenario)
                reason = KNOWN_FAILURES.get((ingress, backend, scenario))
                marks = [pytest.mark.xfail(reason=reason, strict=True)] if reason else []
                params.append(pytest.param(cell, id=cell.id, marks=marks))
    return params
