"""
The gateway serves models by alias from a file it rereads on change: the
dev-to-production backend swap with no app change and no restart.
"""

import base64
import dataclasses
import struct

import pytest
from openai.types import Model

from model_calling.gateway import public_message
from model_calling.aliases import Target, dimension_mismatch
from tests.harness.replay import ReplayTransport
from tests.harness.scenarios import (
    ALL_BACKENDS,
    ALL_SCENARIOS,
    CHAT_COMPLETIONS,
    COHERE_EMBED_MODEL,
    COHERE_MODEL,
    EMBEDDING_SCENARIOS,
    VLLM_URL,
    exchanges_for,
)
from tests.harness.sdk import validate_chat_completion, validate_embedding_response

CHAT_PATH = "/v1/chat/completions"
TEXT = ALL_SCENARIOS["text"]
# The vLLM fixture's model, which the Ollama replay must never be asked for.
VLLM_MODEL = ALL_BACKENDS["vllm"].model_for(TEXT).split("/", 1)[1]
OLLAMA_MODEL = ALL_BACKENDS["ollama"].model_for(TEXT).split("/", 1)[1]
EMBED_DIMENSIONS = 768  # nomic-embed-text, as recorded in fixtures/ollama_embed
COHERE_DIMENSIONS = 1536  # embed-v4.0's default, as in fixtures/cohere_embed


def chat_to(backend: str, model: str, **backend_settings) -> dict:
    """An alias file serving `chat` from `backend`."""
    document = {"models": {"chat": {"backend": backend, "model": model}}}
    if backend_settings:
        document["backends"] = {backend: backend_settings}
    return document


def use_replay(gateway, backend_name: str, exchanges: list) -> ReplayTransport:
    """Answer the backend's prefix from `exchanges`, as `Gateway.use` does,
    without registering any alias."""
    backend = ALL_BACKENDS[backend_name]
    replay = ReplayTransport(exchanges)
    config = gateway.backends.config[backend.config_prefix]
    config.update(backend.gateway_config, transport=replay)
    gateway.replays.append(replay)
    return replay


async def ask_for_chat(gateway) -> dict:
    resp = await gateway.client.post(CHAT_PATH, json=TEXT.chat_completions_body("chat"))
    assert resp.status_code == 200, resp.text
    completion = validate_chat_completion(resp.json())
    assert completion.model == "chat"
    return resp.json()


async def test_editing_the_file_swaps_the_backend_with_no_restart(gateway):
    vllm = use_replay(gateway, "vllm", exchanges_for("vllm", "text", CHAT_COMPLETIONS) * 2)
    ollama = use_replay(gateway, "ollama", exchanges_for("ollama", "text", CHAT_COMPLETIONS))

    gateway.write_config(chat_to("vllm", VLLM_MODEL))
    await ask_for_chat(gateway)
    assert len(vllm.requests) == 1 and not ollama.requests

    # A broken edit (a typo in the backend) leaves `chat` on vLLM.
    gateway.write_config({"models": {"chat": {"backend": "olama", "model": OLLAMA_MODEL}}})
    await ask_for_chat(gateway)
    assert len(vllm.requests) == 2 and not ollama.requests

    gateway.write_config(chat_to("ollama", OLLAMA_MODEL))
    await ask_for_chat(gateway)
    assert len(vllm.requests) == 2 and ollama.requests


async def test_a_named_backend_sets_the_url_and_key(gateway, monkeypatch):
    """The file's URL wins over the gateway's defaults, and the key comes
    from the variable the file names."""
    monkeypatch.setenv("PROD_VLLM_KEY", "prod-key")
    url = "http://vllm.prod.test/v1"
    exchanges = [
        dataclasses.replace(e, url=url + e.url.removeprefix(VLLM_URL))
        for e in exchanges_for("vllm", "text", CHAT_COMPLETIONS)
    ]
    replay = use_replay(gateway, "vllm", exchanges)
    gateway.write_config(
        {
            "backends": {
                "prod": {"provider": "vllm", "base_url": url, "api_key_env": "PROD_VLLM_KEY"}
            },
            "models": {"chat": {"backend": "prod", "model": VLLM_MODEL}},
        }
    )
    await ask_for_chat(gateway)
    assert replay.requests[0].headers["authorization"] == "Bearer prod-key"


async def test_a_cohere_backend_serves_chat_and_embeddings(gateway, monkeypatch):
    """`provider: cohere` in the file: both aliases load, and chat and
    embeddings reach Cohere v2 with the key the file names."""
    monkeypatch.setenv("COHERE_PROD_KEY", "cohere-key")
    replay = use_replay(
        gateway,
        "cohere",
        exchanges_for("cohere", "text", CHAT_COMPLETIONS)
        + exchanges_for("cohere_embed", "embedding"),
    )
    gateway.write_config(
        {
            "backends": {"hosted": {"provider": "cohere", "api_key_env": "COHERE_PROD_KEY"}},
            "models": {
                "chat": {"backend": "hosted", "model": COHERE_MODEL},
                "embed": {
                    "backend": "hosted",
                    "model": COHERE_EMBED_MODEL,
                    "dimensions": COHERE_DIMENSIONS,
                },
            },
        }
    )
    await ask_for_chat(gateway)
    body = EMBEDDING_SCENARIOS["embedding"].body("embed")
    resp = await gateway.client.post("/v1/embeddings", json=body)
    assert resp.status_code == 200, resp.text
    embeddings = validate_embedding_response(resp.json())
    assert embeddings.model == "embed"
    assert {len(d.embedding) for d in embeddings.data} == {COHERE_DIMENSIONS}
    assert [r.url.path for r in replay.requests] == ["/v2/chat", "/v2/embed"]
    assert {r.headers["authorization"] for r in replay.requests} == {"Bearer cohere-key"}


async def test_a_backend_whose_key_variable_is_unset_is_502_without_details(gateway, caplog):
    gateway.write_config(chat_to("vllm", VLLM_MODEL, provider="vllm", api_key_env="UNSET_KEY_VAR"))
    resp = await gateway.client.post(CHAT_PATH, json=TEXT.chat_completions_body("chat"))
    assert resp.status_code == 502, resp.text
    assert resp.json()["error"]["message"] == "model 'chat' is misconfigured on the gateway"
    assert "UNSET_KEY_VAR" in caplog.text


async def test_models_lists_the_aliases_in_file_order(gateway):
    gateway.write_config(
        {
            "models": {
                "chat": {"backend": "vllm", "model": VLLM_MODEL},
                "embed": {"backend": "ollama", "model": "nomic-embed-text"},
            }
        }
    )
    resp = await gateway.client.get("/v1/models")
    assert resp.status_code == 200, resp.text
    models = [Model.model_validate(m) for m in resp.json()["data"]]
    assert [m.id for m in models] == ["chat", "embed"]
    assert {m.owned_by for m in models} == {"model-calling"}


async def test_models_is_empty_without_a_file(gateway):
    resp = await gateway.client.get("/v1/models")
    assert resp.json() == {"object": "list", "data": []}


@pytest.mark.parametrize("declared, status", [(EMBED_DIMENSIONS, 200), (1024, 502), (None, 200)])
async def test_embedding_dimensions_are_checked(gateway, declared, status):
    gateway.use("ollama_embed", "embedding")
    extra = {"dimensions": declared} if declared else {}
    alias = gateway.alias(ALL_BACKENDS["ollama_embed"].gateway_model, **extra)
    body = EMBEDDING_SCENARIOS["embedding"].body(alias)
    resp = await gateway.client.post("/v1/embeddings", json=body)
    assert resp.status_code == status, resp.text
    if status == 200:
        assert validate_embedding_response(resp.json()).model == alias
    else:
        assert resp.json()["error"]["message"] == (
            f"model {alias!r} is declared with {declared} dimensions, "
            f"but its backend returned {EMBED_DIMENSIONS}"
        )


def packed(size: int) -> str:
    return base64.b64encode(struct.pack(f"<{size}f", *([0.5] * size))).decode()


@pytest.mark.parametrize(
    "embedding, declared, refused",
    [
        ([0.1] * 4, 4, False),
        ([0.1] * 4, 3, True),
        (packed(4), 4, False),
        (packed(4), 8, True),
        ("not base64!", 4, False),
    ],
    ids=["list", "list_wrong", "base64", "base64_wrong", "unreadable"],
)
def test_dimension_check_reads_both_encodings(embedding, declared, refused):
    target = Target("e", "ollama/m", None, None, declared)
    response = {"data": [{"embedding": embedding}]}
    assert (dimension_mismatch(response, target) is not None) is refused


@pytest.mark.parametrize(
    "message, expected",
    [
        (
            "vllm returned HTTP 404: The model `gpt-oss-20b` does not exist.",
            "the backend returned HTTP 404: The model `chat` does not exist.",
        ),
        (
            "cannot reach vllm at http://vllm.internal:8000/v1/chat/completions: refused",
            "cannot reach the backend at the backend: refused",
        ),
        ("model 'vllm/gpt-oss-20b' not found", "model 'chat' not found"),
        # A model name inside a longer word or name is left alone.
        ("gpt-oss-20b-instruct is loading", "gpt-oss-20b-instruct is loading"),
        ("VLLM error: overloaded", "the backend error: overloaded"),
    ],
)
def test_public_message_hides_the_backend(message, expected):
    target = Target("chat", "vllm/gpt-oss-20b", "http://vllm.internal:8000/v1", None, None)
    assert public_message(message, target) == expected


async def test_prefix_shaped_alias_serves_alias_name_in_response(gateway):
    """An alias literally named ollama/<model> is served by the gateway,
    and the response's model field is that alias string."""
    # Set up the backend replay
    gateway.use("ollama", "text")
    # Create an alias with a prefix-shaped name
    gateway.write_config(
        {
            "backends": {"local": {"provider": "ollama"}},
            "models": {"ollama/my-model": {"backend": "local", "model": OLLAMA_MODEL}},
        }
    )
    # Make a request using the prefix-shaped alias
    resp = await gateway.client.post(CHAT_PATH, json=TEXT.chat_completions_body("ollama/my-model"))
    assert resp.status_code == 200, resp.text
    completion = validate_chat_completion(resp.json())
    # The response's model field should be the alias, not the backend model
    assert completion.model == "ollama/my-model"
