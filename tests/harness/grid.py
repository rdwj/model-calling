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
    EMBEDDING_BACKENDS,
    EMBEDDING_SCENARIOS,
    GRID_BACKENDS,
    SCENARIOS,
)

# The gateway's HTTP ingresses, and the library API's two call families.
GATEWAY_INGRESSES = ("chat_completions", "gateway_messages")
LIBRARY_INGRESSES = ("library_completion", "library_messages")
INGRESSES = GATEWAY_INGRESSES + LIBRARY_INGRESSES
# Embeddings run over their own backends and scenarios (`embedding_grid`).
GATEWAY_EMBEDDING_INGRESSES = ("gateway_embeddings",)
LIBRARY_EMBEDDING_INGRESSES = ("library_embedding",)

# (ingress, backend, scenario) -> reason, naming the phase that fixes it.
KNOWN_FAILURES: Dict[Tuple[str, str, str], str] = {}


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
    backends: Iterable[str] = GRID_BACKENDS,
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


def embedding_grid(ingresses: Iterable[str]) -> list:
    return grid(EMBEDDING_SCENARIOS, ingresses, EMBEDDING_BACKENDS)
