"""Embeddings response types for the public embedding() API."""

from typing import Literal

from model_calling.types.base import WireModel


class Embedding(WireModel):
    # A string when the caller asked for encoding_format="base64".
    embedding: list[float] | str
    index: int
    object: Literal["embedding"]


class Usage(WireModel):
    prompt_tokens: int
    total_tokens: int


class CreateEmbeddingResponse(WireModel):
    data: list[Embedding]
    model: str
    object: Literal["list"]
    usage: Usage
