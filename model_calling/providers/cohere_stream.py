"""
Cohere v2 chat streaming, and the wire details its stream and its
non-streaming body share: finish reasons, usage, and tool calls.

The stream is server-sent events, each `data:` line one JSON event named by
its `type` (https://docs.cohere.com/reference/chat-stream). Content blocks
(`content-start`/`-delta`/`-end`) and tool calls (`tool-call-start`/`-delta`/
`-end`) count their `index` separately; internal block indexes are one
sequence in order of first appearance, so the parser maps each
(kind, Cohere index) to its own. `tool-plan-delta` events carry no index:
they stream the plan the model writes before its tool calls, which becomes
a text block of its own.
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
    Usage,
)
from model_calling.providers.base import BAD_GATEWAY, StreamParser, new_id

DATA_PREFIX = "data:"
# Cohere's finish_reason -> internal stop reason.
STOP_REASONS = {
    "COMPLETE": "end_turn",
    "STOP_SEQUENCE": "stop_sequence",
    "MAX_TOKENS": "max_tokens",
    "TOOL_CALL": "tool_use",
}
# Finish reasons that mean the generation failed rather than ended.
FAILED_FINISHES = ("ERROR", "TIMEOUT")
THINKING = "thinking"
# Kinds of index space in a Cohere stream, and the tool plan's single block.
_CONTENT, _TOOL_CALL, _PLAN = "content", "tool_call", "plan"


def usage_in(raw: Any) -> Usage | None:
    """Cohere's `usage` (chat) or `meta` (embed) as internal usage. `tokens`
    counts what the model saw and produced; `billed_units`, which leaves out
    Cohere's own prompt template, stands in only when `tokens` is missing."""
    if not isinstance(raw, dict):
        return None
    counts = raw.get("tokens") or raw.get("billed_units") or {}
    cached = raw.get("cached_tokens")
    usage = Usage(
        input_tokens=_count(counts.get("input_tokens")),
        output_tokens=_count(counts.get("output_tokens")),
        cache_read_tokens=_count(cached),
    )
    if usage.input_tokens is None and usage.output_tokens is None:
        return None
    return usage


def _count(value: Any) -> int | None:
    # The spec types token counts as doubles.
    return None if value is None else int(value)


def tool_call_in(raw: dict) -> ToolCall:
    """A `ToolCallV2`, whose `function.arguments` is a JSON string."""
    fn = raw.get("function") or {}
    arguments = fn.get("arguments") or "{}"
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        parsed = None
    call_id = raw.get("id") or new_id("call_")
    if isinstance(parsed, dict):
        return ToolCall(id=call_id, name=fn["name"], input=parsed)
    return ToolCall(id=call_id, name=fn["name"], input={}, raw_arguments=arguments)


def failure_message(finish_reason: str | None, error: str | None) -> str:
    detail = f": {error}" if error else ""
    return f"Cohere generation failed ({finish_reason or 'no finish_reason'}){detail}"


class CohereStreamParser(StreamParser):
    def __init__(self, model: str):
        self.model = model
        self.started = False
        self.done = False
        self.next_index = 0
        # (kind, Cohere index) -> (internal index, block kind) for open blocks.
        self.open: dict[tuple[str, int | None], tuple[int, str]] = {}

    def feed(self, line: str) -> list[StreamEvent]:
        if not line.startswith(DATA_PREFIX):
            # `event:` names repeat the payload's `type`; comments carry nothing.
            return []
        try:
            event = json.loads(line[len(DATA_PREFIX) :])
        except json.JSONDecodeError:
            return [self._error(f"Cohere sent a non-JSON event: {line!r}")]
        kind = event.get("type")
        delta = event.get("delta") or {}
        message = delta.get("message") or {}
        events: list[StreamEvent] = []
        if not self.started and kind != "message-start":
            self.started = True
            events.append(MessageStart(id=new_id(), model=self.model))
        handler = _HANDLERS.get(kind)
        if handler is not None:
            events += handler(self, event, delta, message)
        return events

    def close(self) -> list[StreamEvent]:
        if self.done:
            return []
        return [self._error("Cohere closed the stream before message-end")]

    # --- events -----------------------------------------------------------

    def _message_start(self, event: dict, delta: dict, message: dict) -> list[StreamEvent]:
        if self.started:
            return []
        self.started = True
        return [MessageStart(id=event.get("id") or new_id(), model=self.model)]

    def _content_start(self, event: dict, delta: dict, message: dict) -> list[StreamEvent]:
        content = message.get("content") or {}
        block_kind = THINKING if content.get("type") == THINKING else "text"
        events = self._close_plan()
        events += self._open((_CONTENT, event.get("index")), block_kind)
        events += self._content_text(event, content)
        return events

    def _content_delta(self, event: dict, delta: dict, message: dict) -> list[StreamEvent]:
        key = (_CONTENT, event.get("index"))
        if key not in self.open:
            return [self._error(f"Cohere sent content-delta for unopened block {key[1]}")]
        return self._content_text(event, message.get("content") or {})

    def _content_text(self, event: dict, content: dict) -> list[StreamEvent]:
        index, block_kind = self.open[(_CONTENT, event.get("index"))]
        if block_kind == THINKING:
            text = content.get("thinking")
            delta = ThinkingDelta(thinking=text) if text else None
        else:
            text = content.get("text")
            delta = TextDelta(text=text) if text else None
        return [BlockDelta(index=index, delta=delta)] if delta else []

    def _tool_plan_delta(self, event: dict, delta: dict, message: dict) -> list[StreamEvent]:
        text = message.get("tool_plan")
        events = [] if (_PLAN, None) in self.open else self._open((_PLAN, None), "text")
        if text:
            index, _ = self.open[(_PLAN, None)]
            events.append(BlockDelta(index=index, delta=TextDelta(text=text)))
        return events

    def _tool_call_start(self, event: dict, delta: dict, message: dict) -> list[StreamEvent]:
        raw = message.get("tool_calls") or {}
        fn = raw.get("function") or {}
        call = ToolCall(id=raw.get("id") or new_id("call_"), name=fn.get("name") or "", input={})
        events = self._close_plan()
        events += self._open((_TOOL_CALL, event.get("index")), "tool_call", call)
        events += self._arguments(event, fn)
        return events

    def _tool_call_delta(self, event: dict, delta: dict, message: dict) -> list[StreamEvent]:
        key = (_TOOL_CALL, event.get("index"))
        if key not in self.open:
            return [self._error(f"Cohere sent tool-call-delta for unopened call {key[1]}")]
        return self._arguments(event, (message.get("tool_calls") or {}).get("function") or {})

    def _arguments(self, event: dict, fn: dict) -> list[StreamEvent]:
        arguments = fn.get("arguments")
        if not arguments:
            return []
        index, _ = self.open[(_TOOL_CALL, event.get("index"))]
        return [BlockDelta(index=index, delta=ToolInputDelta(partial_json=arguments))]

    def _content_end(self, event: dict, delta: dict, message: dict) -> list[StreamEvent]:
        return self._stop((_CONTENT, event.get("index")))

    def _tool_call_end(self, event: dict, delta: dict, message: dict) -> list[StreamEvent]:
        return self._stop((_TOOL_CALL, event.get("index")))

    def _message_end(self, event: dict, delta: dict, message: dict) -> list[StreamEvent]:
        self.done = True
        finish = delta.get("finish_reason")
        if finish in FAILED_FINISHES or delta.get("error"):
            return [self._error(failure_message(finish, delta.get("error")))]
        events: list[StreamEvent] = []
        for key in list(self.open):
            events += self._stop(key)
        events.append(
            MessageDelta(stop_reason=STOP_REASONS.get(finish), usage=usage_in(delta.get("usage")))
        )
        events.append(MessageStop())
        return events

    # --- blocks -----------------------------------------------------------

    def _open(self, key: tuple, block_kind: str, call: ToolCall | None = None) -> list[StreamEvent]:
        index = self.next_index
        self.next_index += 1
        self.open[key] = (index, block_kind)
        if call is not None:
            block: Any = call
        elif block_kind == THINKING:
            block = Thinking(thinking="")
        else:
            block = Text(text="")
        return [BlockStart(index=index, block=block)]

    def _stop(self, key: tuple) -> list[StreamEvent]:
        opened = self.open.pop(key, None)
        return [] if opened is None else [BlockStop(index=opened[0])]

    def _close_plan(self) -> list[StreamEvent]:
        return self._stop((_PLAN, None))

    @staticmethod
    def _error(message: str) -> StreamError:
        return StreamError(status=BAD_GATEWAY, message=message)


# Events with nothing to translate (citations, which need documents we never
# send, and debug output) are skipped.
_HANDLERS = {
    "message-start": CohereStreamParser._message_start,
    "content-start": CohereStreamParser._content_start,
    "content-delta": CohereStreamParser._content_delta,
    "content-end": CohereStreamParser._content_end,
    "tool-plan-delta": CohereStreamParser._tool_plan_delta,
    "tool-call-start": CohereStreamParser._tool_call_start,
    "tool-call-delta": CohereStreamParser._tool_call_delta,
    "tool-call-end": CohereStreamParser._tool_call_end,
    "message-end": CohereStreamParser._message_end,
}
