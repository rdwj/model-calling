"""
model-calling: an LLM client library and gateway that speaks OpenAI Chat
Completions and Anthropic Messages over Ollama, vLLM, and OpenAI-compatible
backends.

    import model_calling

    resp = model_calling.completion("ollama/qwen3:4b", [{"role": "user", "content": "hi"}])
    msg = model_calling.messages("vllm/gpt-oss-20b", [...], max_tokens=256)
    vecs = model_calling.embedding("ollama/nomic-embed-text", ["hello"])
"""

from model_calling.errors import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from model_calling.library import (
    Client,
    acompletion,
    aembedding,
    amessages,
    completion,
    embedding,
    messages,
)

__version__ = "0.2.0.dev0"

__all__ = [
    "APIConnectionError",
    "APIError",
    "APIResponseValidationError",
    "APIStatusError",
    "APITimeoutError",
    "AuthenticationError",
    "BadRequestError",
    "Client",
    "InternalServerError",
    "NotFoundError",
    "PermissionDeniedError",
    "RateLimitError",
    "acompletion",
    "aembedding",
    "amessages",
    "completion",
    "embedding",
    "messages",
]
