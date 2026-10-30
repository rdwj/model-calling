"""
model-calling: an LLM client library and gateway that speaks OpenAI Chat
Completions, Anthropic Messages, and OpenAI Responses (stateless) over
Ollama, vLLM, OpenAI-compatible, and Anthropic backends.

    import model_calling

    resp = model_calling.completion("ollama/qwen3:4b", [{"role": "user", "content": "hi"}])
    msg = model_calling.messages("vllm/gpt-oss-20b", [...], max_tokens=256)
    reply = model_calling.responses("ollama/qwen3:4b", "hi", instructions="Be brief.")
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
    UnsupportedParamsError,
)
from model_calling.library import (
    Client,
    acompletion,
    aembedding,
    amessages,
    aresponses,
    completion,
    embedding,
    messages,
    responses,
)

__version__ = "0.3.1"

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
    "UnsupportedParamsError",
    "acompletion",
    "aembedding",
    "amessages",
    "aresponses",
    "completion",
    "embedding",
    "messages",
    "responses",
]
