"""
Gateway settings from the environment: backend URLs and keys.
"""

import dataclasses

import pytest

from model_calling.config import backend_overrides
from tests.harness.gateway import REPLAY_CAPABILITIES
from tests.harness.replay import ReplayTransport
from tests.harness.scenarios import (
    ALL_BACKENDS,
    ALL_SCENARIOS,
    CHAT_COMPLETIONS,
    MESSAGES,
    OLLAMA_URL,
    VLLM_URL,
    exchanges_for,
)

OLLAMA_ENV_URL = "http://ollama.test:11434"
VLLM_ENV_URL = "http://vllm.test:8000/v1"
ANTHROPIC_ENV_URL = "http://anthropic.test/v1"
VLLM_ENV_KEY = "vllm-env-key"

BACKEND_ENV = {
    "OLLAMA_BASE_URL": OLLAMA_ENV_URL,
    "VLLM_BASE_URL": VLLM_ENV_URL,
    "VLLM_API_KEY": VLLM_ENV_KEY,
    "MODEL_CALLING_ANTHROPIC_BASE_URL": ANTHROPIC_ENV_URL,
}


@pytest.mark.parametrize(
    "environ, expected",
    [
        ({}, {}),
        (
            BACKEND_ENV,
            {
                "ollama": {"base_url": OLLAMA_ENV_URL},
                "vllm": {"base_url": VLLM_ENV_URL, "api_key": VLLM_ENV_KEY},
                "anthropic": {"base_url": ANTHROPIC_ENV_URL},
            },
        ),
        ({"VLLM_BASE_URL": f" {VLLM_ENV_URL}/ "}, {"vllm": {"base_url": VLLM_ENV_URL}}),
        ({"OLLAMA_BASE_URL": "", "VLLM_API_KEY": "  "}, {}),
        # The SDK's variable means something else and may point at this gateway.
        ({"ANTHROPIC_BASE_URL": "http://localhost:8000"}, {}),
    ],
)
def test_backend_overrides(environ, expected):
    assert backend_overrides(environ) == expected


@pytest.fixture
def env_gateway(monkeypatch, request):
    """The gateway fixture, built after the backend variables are set."""
    for var, value in BACKEND_ENV.items():
        monkeypatch.setenv(var, value)
    return request.getfixturevalue("gateway")


# (backend, gateway path, form, replay URL the fixture holds, URL the env sets)
ENV_ROUTED = [
    ("ollama", "/v1/chat/completions", CHAT_COMPLETIONS, OLLAMA_URL, OLLAMA_ENV_URL),
    ("vllm", "/v1/chat/completions", CHAT_COMPLETIONS, VLLM_URL, VLLM_ENV_URL),
    ("vllm_messages", "/v1/messages", MESSAGES, VLLM_URL, ANTHROPIC_ENV_URL),
]


@pytest.mark.parametrize("backend_name, path, form, recorded_url, env_url", ENV_ROUTED)
async def test_backend_variables_route_gateway_requests(
    env_gateway, backend_name, path, form, recorded_url, env_url
):
    from model_calling.capability_cache import capability_cache

    backend, scenario = ALL_BACKENDS[backend_name], ALL_SCENARIOS["text"]
    # The recorded exchanges, moved to the host the environment names. Replay
    # answers only an exact URL, so a request anywhere else fails the test.
    exchanges = [
        dataclasses.replace(e, url=env_url + e.url.removeprefix(recorded_url))
        for e in exchanges_for(backend_name, "text", form)
    ]
    assert all(e.url.startswith(env_url) for e in exchanges)
    replay = ReplayTransport(exchanges)
    env_gateway.registry.default_configs[backend.config_prefix]["transport"] = replay
    env_gateway.replays.append(replay)
    model = backend.model_for(scenario)
    capability_cache.set_capability(model, dict(REPLAY_CAPABILITIES))

    body = (
        scenario.messages_body(model) if form == MESSAGES else scenario.chat_completions_body(model)
    )
    response = await env_gateway.client.post(path, json=body)

    assert response.status_code == 200, response.text
    replay.check()
    if backend_name == "vllm":
        assert replay.requests[-1].headers["authorization"] == f"Bearer {VLLM_ENV_KEY}"
