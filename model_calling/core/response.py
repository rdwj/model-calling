"""The internal non-streaming response."""

from typing import Any, Literal

from model_calling.core.base import CoreModel
from model_calling.core.blocks import ContentBlock

# Chat Completions' `content_filter` maps to `refusal`.
StopReason = Literal[
    "end_turn",
    "max_tokens",
    "stop_sequence",
    "tool_use",
    "pause_turn",
    "refusal",
    "context_window_exceeded",
]


class Usage(CoreModel):
    """Token counts. `input_tokens` counts every input token, cached or not,
    and the cache counts are subsets of it. `reasoning_tokens` is a subset of
    `output_tokens`."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None


class Response(CoreModel):
    id: str
    model: str
    content: list[ContentBlock]
    stop_reason: StopReason | None = None
    stop_sequence: str | None = None
    usage: Usage | None = None
    # Anthropic-specific fields: container info and structured refusal details.
    # Kept as dicts since they're format-specific and passed through verbatim.
    stop_details: dict[str, Any] | None = None
    container: dict[str, Any] | None = None
