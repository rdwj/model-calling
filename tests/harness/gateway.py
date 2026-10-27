"""
pytest fixtures that run the FastAPI gateway in-process against replayed backends.

Registered through `pytest_plugins` in tests/conftest.py.
"""

from dataclasses import dataclass

import httpx
import pytest

from tests.harness.replay import Exchange, ReplayTransport, fixture_path
from tests.harness.scenarios import BACKENDS

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
    monkeypatch: pytest.MonkeyPatch

    def use(self, backend_name: str, scenario_name: str) -> ReplayTransport:
        """Point the gateway at `backend_name` and replay its `scenario_name` fixture.

        Returns the transport so tests can inspect what the gateway sent.
        """
        from model_calling.capability_cache import capability_cache

        backend = BACKENDS[backend_name]
        replay = ReplayTransport([Exchange.load(fixture_path(backend_name, scenario_name))])

        self.registry.default_configs[backend.config_prefix].update(backend.gateway_config)
        capability_cache.set_capability(backend.gateway_model, dict(REPLAY_CAPABILITIES))

        # Adapters build their own httpx clients; give any that don't name a
        # transport the replay one. The gateway client passes its own.
        real_init = httpx.AsyncClient.__init__

        def init_with_replay(client, *args, transport=None, **kwargs):
            real_init(client, *args, transport=transport or replay, **kwargs)

        self.monkeypatch.setattr(httpx.AsyncClient, "__init__", init_with_replay)
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
        yield Gateway(client=client, registry=registry, monkeypatch=monkeypatch)
