"""
Internal stream events to Anthropic Messages stream events.

Each event is a dict matching one of the `RawMessageStreamEvent` variants.
The caller writes them as SSE `event:` + `data:` pairs.
"""

from typing import Any

from model_calling.core import (
    BlockDelta,
    BlockStart,
    BlockStop,
    MessageDelta,
    MessageStart,
    MessageStop,
    OpaqueDelta,
    SignatureDelta,
    StreamError,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolInputDelta,
)
from model_calling.formats.anthropic import (
    FORMAT,
    block_from_internal,
    stop_reason_from_internal,
    usage_from_internal,
)

# Status code to Anthropic error type, where we have no backend-specific type.
_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
    504: "timeout_error",
    529: "overloaded_error",
}
# SDK error types the encoder knows.
_KNOWN_ERROR_TYPES = {
    "invalid_request_error",
    "authentication_error",
    "permission_error",
    "not_found_error",
    "rate_limit_error",
    "timeout_error",
    "overloaded_error",
    "api_error",
    "billing_error",
}


class AnthropicStreamEncoder:
    def __init__(self, model: str):
        """`model` is the name the client asked for, which may carry a provider prefix."""
        self.model = model
        self.id: str | None = None

    def encode(self, event: StreamEvent) -> list[dict]:
        if isinstance(event, MessageStart):
            self.id = event.id
            return [self._message_start(event)]
        if self.id is None:
            raise RuntimeError("stream event before message_start")
        if isinstance(event, BlockStart):
            return [self._block_start(event)]
        if isinstance(event, BlockDelta):
            deltas = self._block_delta(event)
            return [
                {"type": "content_block_delta", "index": event.index, "delta": d} for d in deltas
            ]
        if isinstance(event, BlockStop):
            return [{"type": "content_block_stop", "index": event.index}]
        if isinstance(event, MessageDelta):
            return [self._message_delta(event)]
        if isinstance(event, MessageStop):
            return [{"type": "message_stop"}]
        if isinstance(event, StreamError):
            return [self._error(event)]
        return []

    def _message_start(self, event: MessageStart) -> dict:
        usage = event.usage
        input_tokens = 0 if usage is None or usage.input_tokens is None else usage.input_tokens
        message = {
            "id": event.id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": 0,
                "cache_read_input_tokens": None if usage is None else usage.cache_read_tokens,
                "cache_creation_input_tokens": None if usage is None else usage.cache_write_tokens,
            },
        }
        return {"type": "message_start", "message": message}

    def _block_start(self, event: BlockStart) -> dict:
        return {
            "type": "content_block_start",
            "index": event.index,
            "content_block": block_from_internal(event.block),
        }

    def _block_delta(self, event: BlockDelta) -> list[dict]:
        """Return delta dicts for this event; opaque deltas may be dropped."""
        delta = event.delta
        if isinstance(delta, TextDelta):
            return [{"type": "text_delta", "text": delta.text}]
        if isinstance(delta, ThinkingDelta):
            return [{"type": "thinking_delta", "thinking": delta.thinking}]
        if isinstance(delta, SignatureDelta):
            return [{"type": "signature_delta", "signature": delta.signature}]
        if isinstance(delta, ToolInputDelta):
            return [{"type": "input_json_delta", "partial_json": delta.partial_json}]
        if isinstance(delta, OpaqueDelta):
            return [delta.data] if delta.format == FORMAT else []
        return []

    def _message_delta(self, event: MessageDelta) -> dict:
        usage = event.usage
        out_usage: dict[str, Any] = {"output_tokens": 0}
        if usage is not None:
            wire = usage_from_internal(usage)
            out_usage = {
                "output_tokens": wire.get("output_tokens") or 0,
                "input_tokens": wire.get("input_tokens"),
                "cache_read_input_tokens": wire.get("cache_read_input_tokens"),
                "cache_creation_input_tokens": wire.get("cache_creation_input_tokens"),
            }
        return {
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason_from_internal(event.stop_reason),
                "stop_sequence": event.stop_sequence,
            },
            "usage": out_usage,
        }

    def _error(self, event: StreamError) -> dict:
        error_type = event.error_type if event.error_type in _KNOWN_ERROR_TYPES else None
        if error_type is None:
            error_type = _ERROR_TYPES.get(event.status, "api_error")
        return {"type": "error", "error": {"type": error_type, "message": event.message}}
