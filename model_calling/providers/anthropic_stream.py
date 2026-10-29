"""
An Anthropic Messages SSE stream, line by line, to internal events.

Each `data:` line is one `RawMessageStreamEvent`, named by its `type`; the
`event:` lines repeat that name and are skipped. Block indexes pass through,
since Anthropic numbers blocks the way the internal format does. Event and
delta types newer than this module are skipped or kept as `OpaqueDelta`.

A turn with a tool call that ends in `end_turn` (vLLM, when the tool is
forced) ends in `tool_use`, as Anthropic reports it.

`message_delta` usage is cumulative, but a backend may leave out the input
counts it already sent in `message_start`; those are filled in from there.
"""

import json

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
    ToolCall,
    ToolInputDelta,
    Usage,
)
from model_calling.formats import anthropic
from model_calling.providers.base import BAD_GATEWAY, StreamParser, new_id

DATA_PREFIX = "data:"
# Wire usage fields `message_start` reports that `message_delta` may omit.
_INPUT_USAGE = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


class AnthropicStreamParser(StreamParser):
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self.started = False
        self.done = False
        # message_start's usage as sent, to fill in what message_delta omits.
        self.start_wire_usage: dict = {}
        self.called_tool = False

    def feed(self, line: str) -> list[StreamEvent]:
        if not line.startswith(DATA_PREFIX) or self.done:
            return []
        data = line[len(DATA_PREFIX) :].strip()
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            return [self._error(f"{self.provider} sent a non-JSON event: {data!r}")]
        if not isinstance(event, dict):
            return [self._error(f"{self.provider} sent an event that isn't an object: {data!r}")]
        kind = event.get("type")
        if kind == "error":
            err = event.get("error")
            message = err.get("message", err) if isinstance(err, dict) else err
            error_type = err.get("type") if isinstance(err, dict) else None
            return [self._error(f"{self.provider} error: {message}", error_type)]
        if kind == "message_start":
            return self._message_start(event["message"])
        if not self.started:
            if kind == "ping":
                return []
            return [self._error(f"{self.provider} sent {kind!r} before message_start")]
        if kind == "content_block_start":
            block = anthropic.block_to_internal(event["content_block"])
            self.called_tool = self.called_tool or isinstance(block, ToolCall)
            return [BlockStart(index=event["index"], block=block)]
        if kind == "content_block_delta":
            return [BlockDelta(index=event["index"], delta=self._delta(event["delta"]))]
        if kind == "content_block_stop":
            return [BlockStop(index=event["index"])]
        if kind == "message_delta":
            return [self._message_delta(event)]
        if kind == "message_stop":
            self.done = True
            return [MessageStop()]
        # ping, and event types newer than this module.
        return []

    def close(self) -> list[StreamEvent]:
        if self.done:
            return []
        return [self._error(f"{self.provider} closed the stream before message_stop")]

    # --- internals --------------------------------------------------------

    def _message_start(self, message: dict) -> list[StreamEvent]:
        if self.started:
            return [self._error(f"{self.provider} sent a second message_start")]
        self.started = True
        usage = message.get("usage")
        self.start_wire_usage = usage if isinstance(usage, dict) else {}
        return [
            MessageStart(
                id=message.get("id") or new_id(),
                model=message.get("model") or self.model,
                usage=None if usage is None else anthropic.usage_to_internal(usage),
            )
        ]

    @staticmethod
    def _delta(raw: dict):
        kind = raw.get("type")
        if kind == "text_delta":
            return TextDelta(text=raw["text"])
        if kind == "thinking_delta":
            return ThinkingDelta(thinking=raw["thinking"])
        if kind == "signature_delta":
            return SignatureDelta(signature=raw["signature"])
        if kind == "input_json_delta":
            return ToolInputDelta(partial_json=raw["partial_json"])
        return OpaqueDelta(format=anthropic.FORMAT, data=raw)

    def _message_delta(self, event: dict) -> MessageDelta:
        delta = event.get("delta") or {}
        raw_usage = event.get("usage")
        usage = None if raw_usage is None else self._cumulative(raw_usage)
        stop_reason = anthropic.stop_reason_to_internal(delta.get("stop_reason"))
        if stop_reason == "end_turn" and self.called_tool:
            stop_reason = "tool_use"
        return MessageDelta(
            stop_reason=stop_reason,
            stop_sequence=delta.get("stop_sequence"),
            usage=usage,
        )

    def _cumulative(self, raw: dict) -> Usage:
        """Delta usage with each input count it omits taken from message_start,
        merged on the wire so the internal input total counts the cache."""
        merged = dict(raw)
        for name in _INPUT_USAGE:
            if merged.get(name) is None and self.start_wire_usage.get(name) is not None:
                merged[name] = self.start_wire_usage[name]
        return anthropic.usage_to_internal(merged)

    @staticmethod
    def _error(message: str, error_type: str | None = None) -> StreamError:
        return StreamError(status=BAD_GATEWAY, message=message, error_type=error_type)
