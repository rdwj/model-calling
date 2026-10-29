"""
Parametrize tests over (ingress, backend, scenario) cells.

Later phases add ingresses and backends here and delete entries from
KNOWN_FAILURES as they fix them. Known failures are strict xfails: once a
cell starts passing, the run fails until its entry is removed.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import pytest

from tests.harness.scenarios import (
    ALL_BACKENDS,
    ALL_SCENARIOS,
    EMBEDDING_BACKENDS,
    EMBEDDING_SCENARIOS,
    GRID_BACKENDS,
    CHAT_COMPLETIONS,
    MESSAGES,
    SCENARIOS,
)

# The gateway's HTTP ingresses, and the library API's two call families.
GATEWAY_INGRESSES = ("chat_completions", "gateway_messages")
LIBRARY_INGRESSES = ("library_completion", "library_messages")
# The gateway driven by the official `anthropic` and `openai` clients.
SDK_INGRESSES = ("sdk_messages", "sdk_completion")
INGRESSES = GATEWAY_INGRESSES + LIBRARY_INGRESSES + SDK_INGRESSES
# Ingresses that send Chat Completions; they skip scenarios with no such form.
CHAT_COMPLETIONS_INGRESSES = ("chat_completions", "library_completion", "sdk_completion")
# Likewise for Anthropic Messages.
MESSAGES_INGRESSES = ("gateway_messages", "library_messages", "sdk_messages")
# Embeddings run over their own backends and scenarios (`embedding_grid`).
GATEWAY_EMBEDDING_INGRESSES = ("gateway_embeddings",)
LIBRARY_EMBEDDING_INGRESSES = ("library_embedding",)

# (ingress, backend, scenario) -> reason, naming the phase that fixes it.
KNOWN_FAILURES: Dict[Tuple[str, str, str], str] = {
    **{
        (ingress, backend, "parallel_tools"): (
            "gpt-oss-20b on vLLM 0.24 makes one tool call per turn even when asked for "
            "two (backend limitation, found in Phase 5)"
        )
        for ingress in INGRESSES
        for backend in ("vllm", "vllm_messages")
    },
    **{
        (ingress, "openai_compat", "parallel_tools_off"): (
            "mistral-nemo on OpenRouter makes both calls although the request carries "
            "parallel_tool_calls: false (backend limitation, found in Phase 4)"
        )
        for ingress in INGRESSES
    },
}
# Known failures in what we send: replay rejects the request, so these fail
# every test of the cell, and stay marked when `known_failures=False`.
KNOWN_REQUEST_FAILURES: Dict[Tuple[str, str, str], str] = {
    (ingress, backend, "tool_strict"): (
        "a Messages tool's strict is dropped on the way to Chat Completions backends, "
        "even in strict mode (found in Phase 4 review)"
    )
    for ingress in MESSAGES_INGRESSES
    for backend in ("openai_compat", "vllm")
}
KNOWN_FAILURES.update(KNOWN_REQUEST_FAILURES)


@dataclass(frozen=True)
class Cell:
    ingress: str
    backend: str
    scenario: str

    @property
    def form(self) -> str | None:
        """The wire form this cell's ingress sends; None for embeddings."""
        if self.ingress in MESSAGES_INGRESSES:
            return MESSAGES
        if self.ingress in CHAT_COMPLETIONS_INGRESSES:
            return CHAT_COMPLETIONS
        return None

    @property
    def id(self) -> str:
        return f"{self.ingress}-{self.backend}-{self.scenario}"


def grid(
    scenarios: Iterable[str] = tuple(SCENARIOS),
    ingresses: Iterable[str] = INGRESSES,
    backends: Iterable[str] = GRID_BACKENDS,
    known_failures: bool = True,
) -> list:
    """Cells as pytest params. `known_failures=False` leaves KNOWN_FAILURES
    other than KNOWN_REQUEST_FAILURES unmarked, for tests that hold even
    where a cell's checks fail (the gateway still answers as the library
    does)."""
    params = []
    for ingress in ingresses:
        for backend in backends:
            for scenario in scenarios:
                spec = ALL_SCENARIOS[scenario]
                if ingress in CHAT_COMPLETIONS_INGRESSES and not spec.chat_form:
                    continue
                if ingress in MESSAGES_INGRESSES and not spec.messages_form:
                    continue
                cell = Cell(ingress, backend, scenario)
                known = (KNOWN_FAILURES if known_failures else KNOWN_REQUEST_FAILURES).get(
                    (ingress, backend, scenario)
                )
                reason = known or _unsupported(backend, spec)
                marks = [pytest.mark.xfail(reason=reason, strict=True)] if reason else []
                params.append(pytest.param(cell, id=cell.id, marks=marks))
    return params


def _unsupported(backend: str, scenario) -> str | None:
    """Why `backend` can't run `scenario`: a backend limitation, so a strict xfail."""
    check = getattr(ALL_BACKENDS[backend], "unsupported", None)
    return check(scenario) if check else None


def embedding_grid(ingresses: Iterable[str], known_failures: bool = True) -> list:
    return grid(EMBEDDING_SCENARIOS, ingresses, EMBEDDING_BACKENDS, known_failures)
