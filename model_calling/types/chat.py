"""Chat Completions response types for the public completion() API."""

from typing import Any, Literal

from model_calling.types.base import WireModel

FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
ServiceTier = Literal["auto", "default", "flex", "scale", "priority", "fast"]


class PromptTokensDetails(WireModel):
    audio_tokens: int | None = None
    cache_write_tokens: int | None = None
    cached_tokens: int | None = None
    image_tokens: int | None = None
    text_tokens: int | None = None


class CompletionTokensDetails(WireModel):
    accepted_prediction_tokens: int | None = None
    audio_tokens: int | None = None
    reasoning_tokens: int | None = None
    rejected_prediction_tokens: int | None = None
    text_tokens: int | None = None


class CompletionUsage(WireModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails | None = None
    prompt_tokens_details: PromptTokensDetails | None = None


class Function(WireModel):
    arguments: str
    name: str


class ChatCompletionMessageToolCall(WireModel):
    id: str
    function: Function
    type: Literal["function"]


class ChatCompletionMessage(WireModel):
    role: Literal["assistant"]
    content: str | None = None
    refusal: str | None = None
    annotations: list | None = None  # Keep loose; we don't emit these
    audio: Any | None = None  # Keep loose; we don't emit audio details
    tool_calls: list[ChatCompletionMessageToolCall] | None = None
    reasoning: str | None = None  # Non-standard: vLLM/OpenRouter extension


class ChoiceLogprobs(WireModel):
    content: list | None = None  # Deeply nested; keep loose
    refusal: list | None = None


class Choice(WireModel):
    finish_reason: FinishReason
    index: int
    message: ChatCompletionMessage
    logprobs: ChoiceLogprobs | None = None


class ChatCompletion(WireModel):
    """A Chat Completions response from completion()."""

    id: str
    choices: list[Choice]
    created: int
    model: str
    object: Literal["chat.completion"]
    metadata: dict | None = None
    moderation: Any | None = None  # Keep loose; we don't emit moderation
    service_tier: ServiceTier | None = None
    system_fingerprint: str | None = None
    usage: CompletionUsage | None = None


class ChoiceDeltaToolCallFunction(WireModel):
    arguments: str | None = None
    name: str | None = None


class ChoiceDeltaToolCall(WireModel):
    index: int
    id: str | None = None
    function: ChoiceDeltaToolCallFunction | None = None
    type: Literal["function"] | None = None


class ChoiceDelta(WireModel):
    audio: Any | None = None  # Keep loose
    content: str | None = None
    refusal: str | None = None
    role: Literal["developer", "system", "user", "assistant", "tool"] | None = None
    tool_calls: list[ChoiceDeltaToolCall] | None = None
    reasoning: str | None = None  # Non-standard: vLLM/OpenRouter extension


class ChunkChoice(WireModel):
    """A streaming chunk choice (renamed to avoid collision with non-streaming Choice)."""

    delta: ChoiceDelta
    index: int
    finish_reason: FinishReason | None = None
    logprobs: ChoiceLogprobs | None = None


class ChatCompletionChunk(WireModel):
    """A streaming Chat Completions chunk."""

    id: str
    choices: list[ChunkChoice]
    created: int
    model: str
    object: Literal["chat.completion.chunk"]
    moderation: Any | None = None
    obfuscation: str | None = None
    service_tier: ServiceTier | None = None
    system_fingerprint: str | None = None
    usage: CompletionUsage | None = None
