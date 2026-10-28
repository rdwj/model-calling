"""The internal embedding request and response.

Embeddings have no blocks or turns, so these stay small: a batch of inputs
in, one vector per input out, in input order.
"""

from model_calling.core.base import CoreModel


class EmbeddingRequest(CoreModel):
    model: str
    # Always a batch: texts, or token-id arrays for backends that take them.
    input: list[str] | list[list[int]]
    dimensions: int | None = None
    user: str | None = None


class EmbeddingResponse(CoreModel):
    model: str
    vectors: list[list[float]]
    input_tokens: int | None = None
