"""
Gateway settings from the environment: backend URLs and keys, and the admin
token that guards /v1/config/update.
"""

import dataclasses

import httpx
import pytest

from model_calling.config import backend_overrides
from tests.harness.gateway import GATEWAY_BASE_URL, REPLAY_CAPABILITIES
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


ADMIN_TOKEN = "admin-test-token"
CONFIG_UPDATE_PATH = "/v1/config/update"
REDIRECT = {"prefix": "openrouter", "config": {"base_url": "http://attacker.test/v1"}}


def auth(value):
    return {"Authorization": value} if value is not None else {}


def openrouter_url(gateway) -> str:
    return gateway.registry.default_configs["openrouter"]["base_url"]


@pytest.mark.parametrize("authorization", [None, f"Bearer {ADMIN_TOKEN}"])
async def test_config_update_is_absent_without_an_admin_token(gateway, authorization):
    before = openrouter_url(gateway)
    response = await gateway.client.post(
        CONFIG_UPDATE_PATH, json=REDIRECT, headers=auth(authorization)
    )
    # The same answer as a route that doesn't exist.
    unknown = await gateway.client.post("/v1/no-such-endpoint", json=REDIRECT)
    assert (response.status_code, response.json()) == (404, unknown.json())
    assert openrouter_url(gateway) == before


@pytest.mark.parametrize(
    "authorization",
    [
        None,
        "Bearer",
        "Bearer wrong-token",
        f"Bearer {ADMIN_TOKEN}x",
        f"Basic {ADMIN_TOKEN}",
        ADMIN_TOKEN,
        "Bearer ünïcode".encode(),
    ],
)
async def test_config_update_refuses_a_request_without_the_token(
    gateway, monkeypatch, authorization
):
    monkeypatch.setenv("MODEL_CALLING_ADMIN_TOKEN", ADMIN_TOKEN)
    before = openrouter_url(gateway)
    response = await gateway.client.post(
        CONFIG_UPDATE_PATH, json=REDIRECT, headers=auth(authorization)
    )
    assert response.status_code == 403, response.text
    assert openrouter_url(gateway) == before


@pytest.mark.parametrize("scheme", ["Bearer", "bearer"])
async def test_config_update_applies_with_the_token(gateway, monkeypatch, scheme):
    monkeypatch.setenv("MODEL_CALLING_ADMIN_TOKEN", ADMIN_TOKEN)
    response = await gateway.client.post(
        CONFIG_UPDATE_PATH, json=REDIRECT, headers=auth(f"{scheme} {ADMIN_TOKEN}")
    )
    assert response.status_code == 200, response.text
    assert openrouter_url(gateway) == REDIRECT["config"]["base_url"]


@pytest.mark.parametrize("token, status", [(ADMIN_TOKEN, 200), (None, 403)])
async def test_legacy_client_sends_its_admin_token(gateway, monkeypatch, token, status):
    from model_calling import server
    from model_calling.client import ModelCallingClient

    monkeypatch.setenv("MODEL_CALLING_ADMIN_TOKEN", ADMIN_TOKEN)
    client = ModelCallingClient(GATEWAY_BASE_URL, admin_token=token)
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app))
    try:
        if status == 200:
            await client.update_config(REDIRECT["prefix"], REDIRECT["config"])
        else:
            with pytest.raises(httpx.HTTPStatusError) as raised:
                await client.update_config(REDIRECT["prefix"], REDIRECT["config"])
            assert raised.value.response.status_code == status
    finally:
        await client.close()


async def test_vllm_model_listing_sends_the_vllm_key(env_gateway, monkeypatch):
    from model_calling import registry as registry_module

    seen = {}

    def answer(request):
        seen[request.url.host] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": [], "models": []})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        registry_module.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(answer), **kwargs),
    )
    response = await env_gateway.client.get("/v1/models")
    assert response.status_code == 200, response.text
    assert seen == {"vllm.test": f"Bearer {VLLM_ENV_KEY}", "ollama.test": None}
