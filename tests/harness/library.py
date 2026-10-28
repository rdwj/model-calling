"""
pytest fixtures that run the library API against replayed backends.

Registered through `pytest_plugins` in tests/conftest.py.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from model_calling.library import Client
from tests.harness.replay import Exchange, ReplayTransport, fixture_path
from tests.harness.scenarios import BACKENDS


@dataclass
class Library:
    client: Optional[Client] = None
    replays: List[ReplayTransport] = field(default_factory=list)

    def use(self, backend_name: str, scenario_name: str) -> ReplayTransport:
        """A fresh `Client` pointed at `backend_name`, replaying its `scenario_name` fixture.

        Returns the transport so tests can inspect what the client sent.
        """
        if self.client is not None:
            raise RuntimeError("use() a backend once per test")
        backend = BACKENDS[backend_name]
        replay = ReplayTransport([Exchange.load(fixture_path(backend_name, scenario_name))])
        config = {**backend.gateway_config, "transport": replay}
        self.client = Client(config={backend.config_prefix: config})
        self.replays.append(replay)
        return replay


@pytest.fixture
async def library():
    lib = Library()
    yield lib
    try:
        for replay in lib.replays:
            replay.check()
    finally:
        if lib.client is not None:
            await lib.client.aclose()
