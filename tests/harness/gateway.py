"""
pytest fixtures that run the FastAPI gateway in-process against replayed backends.

Registered through `pytest_plugins` in tests/conftest.py.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import pytest
import yaml

from tests.harness.replay import ReplayTransport
from tests.harness.scenarios import ALL_BACKENDS, ALL_SCENARIOS, exchanges_for

GATEWAY_BASE_URL = "http://gateway.test"
# Gateway clients name models by alias. Grid aliases are derived from the
# library's prefixed name but never equal it: a `prefix/model` name must not
# reach the gateway, so it can't be an alias either.
ALIAS_PREFIX = "grid."


def grid_alias(model: str) -> str:
    """The alias the harness gives a library model name: `ollama/qwen3:4b`
    is served as `grid.ollama.qwen3:4b`."""
    return ALIAS_PREFIX + model.replace("/", ".")


# No longer used, kept for reference in recorded fixtures.
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
    backends: object  # The library Client serving ported backends
    # The alias file the gateway reads, and what the harness has put in it.
    config_path: Path
    aliases: Dict[str, dict] = field(default_factory=dict)
    replays: List[ReplayTransport] = field(default_factory=list)

    def alias(self, model: str, **extra) -> str:
        """Serve the library model name `model` (`prefix/name`) under an
        alias, through the prefix's default backend, and return the alias.
        `extra` adds alias settings, such as `dimensions`."""
        name = grid_alias(model)
        prefix, native = model.split("/", 1)
        self.aliases[name] = {"backend": prefix, "model": native, **extra}
        self.write_config({"models": self.aliases})
        return name

    def write_config(self, document: dict) -> None:
        """Replace the alias file as an editor or a ConfigMap update would."""
        staged = self.config_path.with_suffix(".tmp")
        staged.write_text(yaml.safe_dump(document, sort_keys=False))
        # Two writes inside one tick of a coarse filesystem clock could
        # otherwise look like one to the store, which watches mtime.
        self._mtime_ns = max(time.time_ns(), getattr(self, "_mtime_ns", 0) + 1_000_000)
        os.utime(staged, ns=(self._mtime_ns, self._mtime_ns))
        staged.replace(self.config_path)

    def use(
        self, backend_name: str, scenario_name: str, form: Optional[str] = None
    ) -> ReplayTransport:
        """Point the gateway at `backend_name` and replay its `scenario_name`
        fixture, as an ingress sending `form` sees it.

        Returns the transport so tests can inspect what the gateway sent.
        """
        backend = ALL_BACKENDS[backend_name]
        replay = ReplayTransport(exchanges_for(backend_name, scenario_name, form))

        config = self.backends.config[backend.config_prefix]
        config.update(backend.gateway_config, transport=replay)
        model = backend.model_for(ALL_SCENARIOS[scenario_name])
        self.alias(model)
        self.replays.append(replay)
        return replay


@pytest.fixture
async def gateway(monkeypatch, tmp_path):
    """The FastAPI app behind an ASGI transport, with a fresh library client
    and an alias file of its own."""
    from model_calling import server
    from model_calling.aliases import AliasStore

    backends = server.backend_client()
    monkeypatch.setattr(server, "CLIENT", backends)
    config_path = tmp_path / "model-calling.yaml"
    monkeypatch.setattr(server, "MODELS", AliasStore(str(config_path)))
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url=GATEWAY_BASE_URL) as client:
        gw = Gateway(client=client, backends=backends, config_path=config_path)
        yield gw
    await backends.aclose()
    for replay in gw.replays:
        replay.check()
