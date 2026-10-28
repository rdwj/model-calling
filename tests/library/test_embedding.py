"""`Client.embedding` beyond the grid: encodings, strictness, typed failures,
callbacks, and the sync and module-level forms."""

import base64
import json
import struct

import httpx
import pytest

import model_calling
from model_calling.callbacks import Callback
from model_calling.errors import APIResponseValidationError, BadRequestError
from model_calling.library import Client
from tests.harness.replay import Exchange, ReplayTransport, fixture_path
from tests.harness.scenarios import EMBEDDING_BACKENDS, EMBEDDING_INPUTS

OLLAMA = EMBEDDING_BACKENDS["ollama_embed"]
COMPAT = EMBEDDING_BACKENDS["openai_compat_embed"]
EMBED_BODY = {"model": "m", "embeddings": [[1.0, 2.0]], "prompt_eval_count": 3}


def client_for(backend, transport, **kwargs) -> Client:
    config = {backend.config_prefix: {**backend.gateway_config, "transport": transport}}
    return Client(config=config, **kwargs)


def replay(backend) -> ReplayTransport:
    return ReplayTransport([Exchange.load(fixture_path(backend.name, "embedding"))])


def answering(body: dict, sent: list | None = None) -> httpx.MockTransport:
    def handle(request):
        if sent is not None:
            sent.append(json.loads(request.content))
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handle)


@pytest.mark.parametrize("backend", [OLLAMA, COMPAT], ids=lambda b: b.name)
async def test_base64_decodes_to_the_float_vectors(backend):
    async with client_for(backend, replay(backend)) as client:
        floats = await client.aembedding(backend.gateway_model, list(EMBEDDING_INPUTS))
    async with client_for(backend, replay(backend)) as client:
        packed = await client.aembedding(
            backend.gateway_model, list(EMBEDDING_INPUTS), encoding_format="base64"
        )
    for f, b in zip(floats.data, packed.data, strict=True):
        raw = base64.b64decode(b.embedding)
        decoded = struct.unpack(f"<{len(raw) // 4}f", raw)
        assert decoded == pytest.approx(f.embedding, rel=1e-6)


async def test_one_string_is_a_batch_of_one():
    sent = []
    async with client_for(OLLAMA, answering(EMBED_BODY, sent)) as client:
        resp = await client.aembedding("ollama/m", "hi", dimensions=2, user="u")
    # Ollama has no `user`; it only labels the caller, so it goes silently.
    assert sent == [{"model": "m", "input": ["hi"], "dimensions": 2}]
    assert (len(resp.data), resp.usage.prompt_tokens) == (1, 3)


async def test_openai_compatible_backends_get_every_field():
    sent = []
    body = {"data": [{"index": 0, "embedding": [1.0]}], "usage": {"prompt_tokens": 1}}
    async with client_for(COMPAT, answering(body, sent)) as client:
        await client.aembedding("vllm/m", [[1, 2]], user="u", truncate_prompt_tokens=4)
    assert sent == [{"model": "m", "input": [[1, 2]], "user": "u", "truncate_prompt_tokens": 4}]


async def test_ollama_refuses_unknown_fields_unless_dropping():
    sent = []
    async with client_for(OLLAMA, answering(EMBED_BODY, sent)) as client:
        with pytest.raises(BadRequestError, match="cannot honor truncate_prompt_tokens"):
            await client.aembedding("ollama/m", "hi", truncate_prompt_tokens=4)
        assert sent == []
        await client.aembedding("ollama/m", "hi", truncate_prompt_tokens=4, drop_params=True)
    assert sent == [{"model": "m", "input": ["hi"]}]


BAD_REQUESTS = {
    "tokens_on_ollama": ("ollama/m", [1, 2], {}, "text only"),
    "bad_encoding": ("ollama/m", "hi", {"encoding_format": "int8"}, "encoding_format"),
    "empty_input": ("ollama/m", [], {}, "input must be"),
}


@pytest.mark.parametrize("model, input, params, match", BAD_REQUESTS.values(), ids=BAD_REQUESTS)
async def test_requests_the_backend_cannot_take_are_bad_requests(model, input, params, match):
    async with client_for(OLLAMA, answering(EMBED_BODY)) as client:
        with pytest.raises(BadRequestError, match=match) as err:
            await client.aembedding(model, input, **params)
    assert err.value.provider == "ollama"


UNREADABLE = {
    "count_mismatch": (EMBED_BODY, "1 embeddings for 2 inputs"),
    "no_embeddings": ({"model": "m"}, "could not read"),
}


@pytest.mark.parametrize("body, match", UNREADABLE.values(), ids=UNREADABLE)
async def test_unreadable_responses_are_typed(body, match):
    async with client_for(OLLAMA, answering(body)) as client:
        with pytest.raises(APIResponseValidationError, match=match):
            await client.aembedding("ollama/m", ["a", "b"])


class EmbeddingRecorder(Callback):
    def __init__(self):
        self.log = []

    async def on_embedding_request_start(self, model, input_texts, **kwargs):
        self.log.append(("start", model, input_texts, kwargs))

    async def on_embedding_request_end(self, model, input_texts, response, **kwargs):
        self.log.append(("end", model, input_texts, kwargs))

    async def on_error(self, model, input_texts, error, **kwargs):
        self.log.append(("error", model, input_texts, type(error).__name__))


async def test_callbacks_see_each_embedding_call():
    cb = EmbeddingRecorder()
    async with client_for(OLLAMA, answering(EMBED_BODY), callbacks=[cb]) as client:
        await client.aembedding("ollama/m", "hi", dimensions=2)
        with pytest.raises(APIResponseValidationError):
            await client.aembedding("ollama/m", ["a", "b"])
    assert cb.log == [
        ("start", "m", ["hi"], {"dimensions": 2}),
        ("end", "m", ["hi"], {"dimensions": 2}),
        ("start", "m", ["a", "b"], {}),
        ("error", "m", ["a", "b"], "APIResponseValidationError"),
    ]


def test_sync_embedding():
    with client_for(OLLAMA, replay(OLLAMA)) as client:
        resp = client.embedding(OLLAMA.gateway_model, list(EMBEDDING_INPUTS))
    assert len(resp.data) == len(EMBEDDING_INPUTS)


async def test_module_level_functions_use_the_default_client(monkeypatch):
    client = client_for(OLLAMA, answering(EMBED_BODY))
    monkeypatch.setattr(model_calling.library, "_default", client)
    resp = await model_calling.aembedding("ollama/m", "hi")
    assert resp.data[0].embedding == [1.0, 2.0]
    await client.aclose()
