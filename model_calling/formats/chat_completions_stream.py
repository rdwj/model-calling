"""
Internal stream events to Chat Completions chunks.

Every chunk of one stream shares the response id from `MessageStart`. The
caller writes each chunk as an SSE `data:` line and ends with `[DONE]`.
"""

import time
from typing import Any

from model_calling.core import (
    BlockDelta,
    BlockStart,
    MessageDelta,
    MessageStart,
    MessageStop,
    StreamError,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolInputDelta,
)
from model_calling.formats.chat_completions import (
    FINISH_REASONS,
    tool_call_out,
    error_body,
    usage_out,
)


class ChatCompletionsStreamEncoder:
    def __init__(self, model: str, include_usage: bool = False):
        """`model` is the name the client asked for, which may carry a provider prefix."""
        self.model = model
        self.include_usage = include_usage
        self.created = int(time.time())
        self.id: str | None = None
        self.usage = None
        # Internal block index -> Chat Completions tool_calls index.
        self.tool_index: dict[int, int] = {}

    def _chunk(self, delta: dict, finish_reason: str | None = None) -> dict:
        if self.id is None:
            raise RuntimeError("stream event before message_start")
        return {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    def encode(self, event: StreamEvent) -> list[dict]:
        if isinstance(event, MessageStart):
            self.id = event.id
            return [self._chunk({"role": "assistant", "content": ""})]
        if isinstance(event, BlockStart) and isinstance(event.block, ToolCall):
            index = self.tool_index[event.index] = len(self.tool_index)
            return [self._chunk({"tool_calls": [tool_call_out(event.block, index)]})]
        if isinstance(event, BlockDelta):
            return self._delta(event)
        if isinstance(event, MessageDelta):
            if event.usage is not None:
                self.usage = event.usage
            return [self._chunk({}, FINISH_REASONS.get(event.stop_reason, "stop"))]
        if isinstance(event, MessageStop):
            if self.include_usage and self.usage is not None:
                return [{**self._chunk({}), "choices": [], "usage": usage_out(self.usage)}]
            return []
        if isinstance(event, StreamError):
            return [error_body(event.status, event.message)]
        return []

    def _delta(self, event: BlockDelta) -> list[dict]:
        delta = event.delta
        out: dict[str, Any]
        if isinstance(delta, TextDelta):
            out = {"content": delta.text}
        elif isinstance(delta, ThinkingDelta):
            out = {"reasoning": delta.thinking}
        elif isinstance(delta, ToolInputDelta):
            index = self.tool_index[event.index]
            out = {"tool_calls": [{"index": index, "function": {"arguments": delta.partial_json}}]}
        else:
            # Signatures and opaque deltas have no Chat Completions form.
            return []
        return [self._chunk(out)]
