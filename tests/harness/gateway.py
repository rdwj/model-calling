"""
pytest fixtures that run the FastAPI gateway in-process against replayed backends.

Registered through `pytest_plugins` in tests/conftest.py.
"""

from dataclasses import dataclass, field
from typing import List

import httpx
import pytest

from tests.harness.replay import ReplayTransport
from tests.harness.scenarios import ALL_BACKENDS, ALL_SCENARIOS, exchanges_for

GATEWAY_BASE_URL = "http://gateway.test"

# Seeded as verified so the registry never schedules a live capability probe.
REPLAY_CAPABILITIES = {
    "supports_function_call": True,
    "supports_streaming": True,
    "supports_system_messages": True,
    "supports_vision": False,
    "supports_json_mode": True,
    "supports_reasoning": False,
    "thinking_style": "none",
    "max_tokens": None,
    "verified": True,
}


@dataclass
class Gateway:
    client: httpx.AsyncClient
    registry: object
    replays: List[ReplayTransport] = field(default_factory=list)

    def use(self, backend_name: str, scenario_name: str) -> ReplayTransport:
        """Point the gateway at `backend_name` and replay its `scenario_name` fixture.

        Returns the transport so tests can inspect what the gateway sent.
        """
        from model_calling.capability_cache import capability_cache

        backend = ALL_BACKENDS[backend_name]
        replay = ReplayTransport(exchanges_for(backend_name, scenario_name))

        config = self.registry.default_configs[backend.config_prefix]
        config.update(backend.gateway_config, transport=replay)
        model = backend.model_for(ALL_SCENARIOS[scenario_name])
        capability_cache.set_capability(model, dict(REPLAY_CAPABILITIES))
        self.replays.append(replay)
        return replay


@pytest.fixture
async def gateway(monkeypatch):
    """The FastAPI app behind an ASGI transport, with a fresh model registry."""
    from model_calling import server
    from model_calling.registry import ModelRegistry

    registry = ModelRegistry()
    monkeypatch.setattr(server, "registry", registry)
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url=GATEWAY_BASE_URL) as client:
        gw = Gateway(client=client, registry=registry)
        yield gw
    await registry.client.aclose()
    for replay in gw.replays:
        replay.check()
