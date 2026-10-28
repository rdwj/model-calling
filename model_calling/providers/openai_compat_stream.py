"""
A Chat Completions SSE stream, line by line, to internal events.

Reasoning, text, and each tool call become separate blocks, in the order
they first appear. A tool call's first fragment names it (and usually its
id); later fragments carry slices of its arguments. Tool calls must arrive
one after another: a fragment for a call whose block has already closed is
an error rather than a silently reordered stream.

`finish_reason` and `usage` can arrive in different chunks (usage last, with
empty `choices`, when `stream_options.include_usage` is set), so the closing
events wait for `[DONE]`.
"""

import json
from typing import Any

from model_calling.core import (
    BlockDelta,
    BlockStart,
    BlockStop,
    MessageDelta,
    MessageStart,
    MessageStop,
    StreamError,
    StreamEvent,
    Text,
    TextDelta,
    Thinking,
    ThinkingDelta,
    ToolCall,
    ToolInputDelta,
)
from model_calling.formats.chat_completions import REASONING_FIELDS, STOP_REASONS, usage_in
from model_calling.providers.base import BAD_GATEWAY, StreamParser, new_id

DATA_PREFIX = "data:"
DONE = "[DONE]"


class ChatCompletionsStreamParser(StreamParser):
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self.started = False
        self.index = -1
        # Kind of the open block ("thinking", "text", or "tool"), if any.
        self.open: str | None = None
        # Chat Completions tool_calls index -> internal block index.
        self.tool_blocks: dict[int, int] = {}
        self.finish_reason: str | None = None
        self.usage = None
        self.done = False

    def feed(self, line: str) -> list[StreamEvent]:
        # Comments (": OPENROUTER PROCESSING") and other SSE fields carry nothing.
        if not line.startswith(DATA_PREFIX) or self.done:
            return []
        data = line[len(DATA_PREFIX) :].strip()
        if data == DONE:
            return self._finish()
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            return [self._error(f"{self.provider} sent a non-JSON chunk: {data!r}")]
        if not isinstance(chunk, dict):
            return [self._error(f"{self.provider} sent a chunk that isn't an object: {data!r}")]
        if chunk.get("error"):
            err = chunk["error"]
            message = err.get("message", err) if isinstance(err, dict) else err
            return [self._error(f"{self.provider} error: {message}")]
        events: list[StreamEvent] = []
        if not self.started:
            self.started = True
            events.append(
                MessageStart(id=chunk.get("id") or new_id(), model=chunk.get("model", self.model))
            )
        if chunk.get("usage"):
            self.usage = usage_in(chunk["usage"])
        for choice in chunk.get("choices") or []:
            if choice.get("index", 0) != 0:
                continue
            events += self._delta(choice.get("delta") or {})
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
                events += self._close()
        return events

    def close(self) -> list[StreamEvent]:
        if self.done:
            return []
        return [self._error(f"{self.provider} closed the stream before it finished")]

    # --- internals --------------------------------------------------------

    def _delta(self, delta: dict) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        reasoning = next((delta[f] for f in REASONING_FIELDS if delta.get(f)), None)
        if reasoning:
            events += self._into("thinking", ThinkingDelta(thinking=reasoning))
        if delta.get("content"):
            events += self._into("text", TextDelta(text=delta["content"]))
        for fragment in delta.get("tool_calls") or []:
            events += self._tool_fragment(fragment)
        return events

    def _tool_fragment(self, fragment: dict) -> list[StreamEvent]:
        tc_index = fragment.get("index", 0)
        fn = fragment.get("function") or {}
        events: list[StreamEvent] = []
        if tc_index not in self.tool_blocks:
            events += self._close()
            self.index += 1
            self.open = "tool"
            self.tool_blocks[tc_index] = self.index
            call = ToolCall(
                id=fragment.get("id") or new_id("call_"), name=fn.get("name") or "", input={}
            )
            events.append(BlockStart(index=self.index, block=call))
        elif self.tool_blocks[tc_index] != self.index or self.open != "tool":
            return [self._error(f"{self.provider} interleaved tool call arguments")]
        if fn.get("arguments"):
            events.append(
                BlockDelta(index=self.index, delta=ToolInputDelta(partial_json=fn["arguments"]))
            )
        return events

    def _into(self, kind: str, delta: Any) -> list[StreamEvent]:
        """Route a delta into an open block of `kind`, opening one if needed."""
        events: list[StreamEvent] = []
        if self.open != kind:
            events += self._close()
            self.index += 1
            self.open = kind
            block = Thinking(thinking="") if kind == "thinking" else Text(text="")
            events.append(BlockStart(index=self.index, block=block))
        events.append(BlockDelta(index=self.index, delta=delta))
        return events

    def _close(self) -> list[StreamEvent]:
        if self.open is None:
            return []
        self.open = None
        return [BlockStop(index=self.index)]

    def _finish(self) -> list[StreamEvent]:
        if not self.started:
            return [self._error(f"{self.provider} ended the stream without a response")]
        self.done = True
        stop_reason = STOP_REASONS.get(self.finish_reason)
        if self.tool_blocks and stop_reason == "end_turn":
            stop_reason = "tool_use"
        return self._close() + [
            MessageDelta(stop_reason=stop_reason, usage=self.usage),
            MessageStop(),
        ]

    @staticmethod
    def _error(message: str) -> StreamError:
        return StreamError(status=BAD_GATEWAY, message=message)
