"""
OpenAI Embeddings ingress: `/v1/embeddings` requests in, responses out.

`encoding_format` is honored here rather than by the backend: every
provider asks for floats, and `base64` is applied on the way out, as
little-endian float32 (what the `openai` SDK decodes).
"""

import base64
import struct
from typing import Any

from model_calling.core.embeddings import EmbeddingRequest, EmbeddingResponse
from model_calling.formats import TranslationError

FORMAT = "openai_embeddings"
ENCODING_FORMATS = ("float", "base64")
_MANAGED_FIELDS = ("model", "input", "dimensions", "user", "encoding_format")


def _is_tokens(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(t, int) and not isinstance(t, bool) for t in value
    )


def _batch(raw: Any) -> list[str] | list[list[int]]:
    """`input` as a non-empty batch: one string or token array becomes a list of one."""
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list) and raw:
        if all(isinstance(x, str) for x in raw):
            return raw
        if _is_tokens(raw):
            return [raw]
        if all(_is_tokens(x) and x for x in raw):
            return raw
    raise TranslationError(
        "input must be a string, a non-empty list of strings, a token array, "
        f"or a list of token arrays, not {raw!r:.100}"
    )


def request_to_internal(body: dict) -> EmbeddingRequest:
    if not isinstance(body, dict):
        raise TranslationError("an embeddings request body must be a JSON object")
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise TranslationError("model is required")
    if "input" not in body:
        raise TranslationError("input is required")
    dimensions = body.get("dimensions")
    if dimensions is not None and (
        not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions < 1
    ):
        raise TranslationError(f"dimensions must be a positive integer, not {dimensions!r}")
    user = body.get("user")
    if user is not None and not isinstance(user, str):
        raise TranslationError(f"user must be a string, not {user!r}")
    encoding_format(body)
    extras = {k: v for k, v in body.items() if k not in _MANAGED_FIELDS}
    return EmbeddingRequest(
        model=model,
        input=_batch(body["input"]),
        dimensions=dimensions,
        user=user,
        extras={FORMAT: extras} if extras else {},
    )


def encoding_format(body: dict) -> str:
    """The client's `encoding_format`, `float` when unset."""
    value = body.get("encoding_format") or "float"
    if value not in ENCODING_FORMATS:
        raise TranslationError(
            f"encoding_format must be one of {', '.join(ENCODING_FORMATS)}, not {value!r}"
        )
    return value


def _encode(vector: list[float], fmt: str) -> list[float] | str:
    if fmt == "float":
        return vector
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")


def response_from_internal(response: EmbeddingResponse, model: str, fmt: str = "float") -> dict:
    """An OpenAI `CreateEmbeddingResponse` body. `model` is the client's id."""
    tokens = response.input_tokens or 0
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": _encode(vector, fmt), "index": i}
            for i, vector in enumerate(response.vectors)
        ],
        "model": model,
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }
