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
