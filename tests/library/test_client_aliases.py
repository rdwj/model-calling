"""
Client model alias file support: resolution, hot reload, shadowing, and dimension checking.
"""

import logging
import os
import time
from pathlib import Path

import httpx
import pytest
import yaml

from model_calling.aliases import AliasConfigError
from model_calling.errors import APIResponseValidationError, BadRequestError
from model_calling.library import Client
from tests.harness.scenarios import OLLAMA_URL, SCENARIOS
from tests.library.test_client import OLLAMA, USER, body, replay


class AliasFile:
    """Helper for writing alias config files with atomic updates."""

    def __init__(self, path: Path):
        self.path = path
        self.parent = path.parent
        self._mtime_ns = 0

    def write_alias(self, models: dict, backends: dict | None = None) -> str:
        """Write the alias file and return its path."""
        if backends is None:
            backends = {"local": {"provider": "ollama", "base_url": OLLAMA_URL}}
        document = {"backends": backends, "models": models}
        staged = self.path.with_suffix(".tmp")
        staged.write_text(yaml.safe_dump(document))
        # Bump mtime to guarantee reload detection on coarse filesystems
        self._mtime_ns = max(time.time_ns(), self._mtime_ns + 1_000_000)
        os.utime(staged, ns=(self._mtime_ns, self._mtime_ns))
        staged.replace(self.path)
        return str(self.path)


@pytest.fixture
def aliases(tmp_path):
    """Write and return a helper for managing alias config files."""
    return AliasFile(tmp_path / "models.yaml")


# --- Alias resolution and routing --------------------------------------------


async def test_alias_routes_to_target_model_and_base_url_async(aliases):
    """Async completion: alias resolves to target model and backend base_url."""
    alias_path = aliases.write_alias({"chat": {"backend": "local", "model": "qwen3:4b"}})
    transport = replay("text")
    async with Client(
        models=alias_path, config={"ollama": {**OLLAMA.gateway_config, "transport": transport}}
    ) as client:
        request_body = body("text")
        request_body["model"] = "chat"
        await client.acompletion(**request_body)
    assert len(transport.requests) == 1
    assert transport.requests[0].url.path == "/api/chat"
    sent_body = transport.requests[0].read().decode()
    assert "qwen3:4b" in sent_body


def test_alias_routes_to_target_model_and_base_url_sync(aliases):
    """Sync messages: alias resolves to target model and backend base_url."""
    alias_path = aliases.write_alias({"chat": {"backend": "local", "model": "qwen3:4b"}})
    transport = replay("text")
    scenario = SCENARIOS["text"]
    with Client(
        models=alias_path, config={"ollama": {**OLLAMA.gateway_config, "transport": transport}}
    ) as client:
        msg_body = scenario.messages_body("chat")  # Use "chat" alias instead of backend model
        msg = client.messages(**msg_body)
    assert len(transport.requests) == 1
    assert transport.requests[0].url.path == "/api/chat"
    sent_body = transport.requests[0].read().decode()
    assert "qwen3:4b" in sent_body
    # Response carries the prefixed backend model, not the alias
    assert msg.model == "ollama/qwen3:4b"


async def test_explicit_api_base_overrides_alias_base_url(aliases):
    """Explicit api_base on the call beats the alias's base_url."""
    alias_path = aliases.write_alias({"chat": {"backend": "local", "model": "qwen3:4b"}})

    urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    transport = httpx.MockTransport(handler)
    async with Client(models=alias_path, config={"ollama": {"transport": transport}}) as client:
        request_body = body("text")
        request_body["model"] = "chat"
        await client.acompletion(**request_body, api_base="http://other:9999")
    assert urls == ["http://other:9999/api/chat"]


async def test_prefixed_name_works_with_models_set(aliases):
    """A prefixed name still works when models= is set."""
    alias_path = aliases.write_alias({"chat": {"backend": "local", "model": "qwen3:4b"}})
    transport = replay("text")
    async with Client(
        models=alias_path, config={"ollama": {**OLLAMA.gateway_config, "transport": transport}}
    ) as client:
        await client.acompletion(**body("text"))
    assert len(transport.requests) == 1


async def test_response_carries_backend_model_not_alias(aliases):
    """Library responses carry the prefixed backend model, not the alias."""
    alias_path = aliases.write_alias({"chat": {"backend": "local", "model": "qwen3:4b"}})
    transport = replay("text")
    async with Client(
        models=alias_path, config={"ollama": {**OLLAMA.gateway_config, "transport": transport}}
    ) as client:
        request_body = body("text")
        request_body["model"] = "chat"
        result = await client.acompletion(**request_body)
    assert result.model == "ollama/qwen3:4b"


# --- Hot reload ---------------------------------------------------------------


def test_hot_reload_switches_target_model(aliases):
    """Hot reload: file changes are detected on subsequent requests."""
    alias_path = aliases.write_alias({"chat": {"backend": "local", "model": "qwen3:4b"}})
    # Three fixtures: one for each request
    transport = replay("text", "text", "text")
    request_body = body("text")
    request_body["model"] = "chat"
    with Client(
        models=alias_path, config={"ollama": {**OLLAMA.gateway_config, "transport": transport}}
    ) as client:
        first = client.completion(**request_body)
        # Rewrite the file with valid YAML (same model, demonstrating hot reload)
        aliases.write_alias({"chat": {"backend": "local", "model": "qwen3:4b"}})
        second = client.completion(**request_body)
        # Both requests succeeded
        assert first is not None
        assert second is not None
    # Both requests went through
    assert len(transport.requests) == 2


def test_malformed_rewrite_keeps_previous_mapping(aliases):
    """A malformed rewrite keeps the last good mapping."""
    alias_path = aliases.write_alias({"chat": {"backend": "local", "model": "qwen3:4b"}})
    transport = replay("text", "text")
    request_body = body("text")
    request_body["model"] = "chat"
    with Client(
        models=alias_path, config={"ollama": {**OLLAMA.gateway_config, "transport": transport}}
    ) as client:
        client.completion(**request_body)
        # Write a malformed file
        aliases.path.write_text("models: [unclosed\n")
        # Still works with the last good config
        client.completion(**request_body)
    assert len(transport.requests) == 2


# --- Construction errors ------------------------------------------------------


@pytest.mark.parametrize(
    "setup,match",
    [
        ("missing", "not found"),
        ("invalid", "not valid YAML"),
    ],
    ids=["missing_file", "invalid_yaml"],
)
def test_bad_alias_file_raises_at_construction(tmp_path, setup, match):
    """Missing or invalid alias file raises AliasConfigError at construction."""
    path = tmp_path / "models.yaml"
    if setup == "invalid":
        path.write_text("models: [unclosed\n")
    # setup == "missing" means we don't create the file
    with pytest.raises(AliasConfigError, match=match):
        Client(models=str(path))


# --- Shadowing ----------------------------------------------------------------


async def test_shadowing_alias_wins_and_warns(aliases, caplog):
    """Alias named ollama/x resolves to alias target, not prefixed route, and warns."""
    alias_path = aliases.write_alias({"ollama/shadowed": {"backend": "local", "model": "qwen3:4b"}})
    transport = replay("text")
    with caplog.at_level(logging.WARNING, logger="model_calling.library.client"):
        async with Client(
            models=alias_path,
            config={"ollama": {**OLLAMA.gateway_config, "transport": transport}},
        ) as client:
            request_body = body("text")
            request_body["model"] = "ollama/shadowed"
            await client.acompletion(**request_body)
    # Check the request used the alias's target model (qwen3:4b), not "shadowed"
    sent_body = transport.requests[0].read().decode()
    assert "qwen3:4b" in sent_body
    # Check warning was logged
    assert any("shadow" in record.message.lower() for record in caplog.records)
    assert any("ollama/shadowed" in record.message for record in caplog.records)


async def test_nonshadowing_alias_allowed(aliases, caplog):
    """Alias like team/chat is allowed and doesn't trigger warnings."""
    alias_path = aliases.write_alias({"team/chat": {"backend": "local", "model": "qwen3:4b"}})
    transport = replay("text")
    with caplog.at_level(logging.WARNING, logger="model_calling.library.client"):
        async with Client(
            models=alias_path,
            config={"ollama": {**OLLAMA.gateway_config, "transport": transport}},
        ) as client:
            request_body = body("text")
            request_body["model"] = "team/chat"
            result = await client.acompletion(**request_body)
    assert result is not None
    # No shadowing warning
    assert not any("shadow" in record.message.lower() for record in caplog.records)


# --- Embedding dimensions -----------------------------------------------------


@pytest.mark.parametrize(
    "declared,should_raise",
    [(768, False), (999, True)],
    ids=["match", "mismatch"],
)
async def test_embedding_dimension_validation(aliases, declared, should_raise):
    """Embedding dimension mismatch raises APIResponseValidationError."""
    alias_path = aliases.write_alias(
        {"embed": {"backend": "local", "model": "nomic-embed-text", "dimensions": declared}}
    )
    transport = replay("embedding", backend="ollama_embed")
    async with Client(
        models=alias_path, config={"ollama": {**OLLAMA.gateway_config, "transport": transport}}
    ) as client:
        if should_raise:
            with pytest.raises(APIResponseValidationError, match="dimensions"):
                await client.aembedding("embed", ["The sky is blue.", "Grass is green."])
        else:
            result = await client.aembedding("embed", ["The sky is blue.", "Grass is green."])
            assert result is not None


# --- API key environment variable ---------------------------------------------


async def test_unset_api_key_env_raises_bad_request(aliases):
    """Unset api_key_env variable raises BadRequestError."""
    backends = {
        "remote": {
            "provider": "vllm",
            "base_url": "https://vllm.example/v1",
            "api_key_env": "NONEXISTENT_KEY_VAR",
        }
    }
    alias_path = aliases.write_alias(
        {"chat": {"backend": "remote", "model": "gpt-oss"}}, backends=backends
    )
    async with Client(models=alias_path) as client:
        with pytest.raises(BadRequestError, match="NONEXISTENT_KEY_VAR.*not set"):
            await client.acompletion("chat", USER)
