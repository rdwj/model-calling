"""
Internal stream events to OpenAI Responses stream events.

A stream is `response.created` and `response.in_progress`, then for each
output item `response.output_item.added`, its own events, and
`response.output_item.done`, then `response.completed` (or
`response.incomplete`) carrying the whole response. Every event has a
`sequence_number`, counting from 0. The caller writes each as an SSE
`event:` + `data:` pair. Item ids and output match what
`responses.response_from_internal` returns for the same response.
"""

import time
from dataclasses import dataclass
from typing import Any

from model_calling.core import (
    BlockDelta,
    BlockStart,
    BlockStop,
    MessageDelta,
    MessageStart,
    MessageStop,
    Request,
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
from model_calling.formats.responses import (
    item_id,
    item_kind,
    output_item,
    output_text,
    response_body,
    status_for,
    stream_error_event,
    summary_text,
)

# The terminal event for each final status.
_DONE_EVENTS = {"completed": "response.completed", "incomplete": "response.incomplete"}


@dataclass
class _Item:
    """An output item being streamed, and the internal block it comes from."""

    output_index: int
    id: str
    block: Any
    # Text, thinking, or tool arguments received so far.
    text: str = ""

    def final(self) -> Any:
        """The internal block, filled in from the deltas."""
        if isinstance(self.block, Text):
            return self.block.model_copy(update={"text": self.text})
        if isinstance(self.block, Thinking):
            return self.block.model_copy(update={"thinking": self.text})
        return self.block.model_copy(update={"input": {}, "raw_arguments": self.text})


class ResponsesStreamEncoder:
    def __init__(self, model: str, request: Request):
        """`model` is the name the client asked for; `request` is the
        internal form of the client's request, whose settings the
        response echoes."""
        self.model = model
        self.request = request
        self.created_at = int(time.time())
        self.id: str | None = None
        self.sequence = 0
        # Internal block index -> the output item it streams as.
        self.items: dict[int, _Item] = {}
        # Output index -> the finished item.
        self.done: dict[int, dict] = {}
        self.stop_reason: str | None = None
        self.usage: Usage | None = None

    def _event(self, kind: str, **fields: Any) -> dict:
        event = {"type": kind, "sequence_number": self.sequence, **fields}
        self.sequence += 1
        return event

    def _response(self, status: str) -> dict:
        return response_body(
            self.id,
            self.model,
            self.request,
            status=status,
            created_at=self.created_at,
            output=self._output(status),
            stop_reason=self.stop_reason,
            usage=self.usage,
        )

    def _output(self, status: str) -> list[dict]:
        if status == "in_progress":
            return []
        output = [self.done[i] for i in sorted(self.done)]
        if status == "incomplete" and output:
            # The last item is the one cut short.
            output[-1] = {**output[-1], "status": "incomplete"}
        return output

    def encode(self, event: StreamEvent) -> list[dict]:
        if isinstance(event, StreamError):
            return [stream_error_event(event.status, event.message, self._take_sequence())]
        if isinstance(event, MessageStart):
            self.id = event.id
            self.usage = event.usage
            return [
                self._event("response.created", response=self._response("in_progress")),
                self._event("response.in_progress", response=self._response("in_progress")),
            ]
        if self.id is None:
            raise RuntimeError("stream event before message_start")
        if isinstance(event, BlockStart):
            return self._start(event)
        if isinstance(event, BlockDelta):
            return self._delta(event)
        if isinstance(event, BlockStop):
            return self._stop(event)
        if isinstance(event, MessageDelta):
            self.stop_reason = event.stop_reason
            if event.usage is not None:
                self.usage = event.usage
            return []
        if isinstance(event, MessageStop):
            status = status_for(self.stop_reason)
            return [self._event(_DONE_EVENTS[status], response=self._response(status))]
        return []

    def _take_sequence(self) -> int:
        self.sequence += 1
        return self.sequence - 1

    def _start(self, event: BlockStart) -> list[dict]:
        kind = item_kind(event.block)
        index = len(self.items)
        added = output_item(event.block, kind and item_id(kind, self.id, index), "in_progress")
        if added is None:
            return []
        item = self.items[event.index] = _Item(index, added["id"], event.block)
        place = {"item_id": item.id, "output_index": index}
        if kind == "message":
            added["content"] = []
            part = output_text("")
            return [
                self._event("response.output_item.added", output_index=index, item=added),
                self._event("response.content_part.added", **place, content_index=0, part=part),
            ]
        if kind == "reasoning":
            added["summary"] = []
            return [
                self._event("response.output_item.added", output_index=index, item=added),
                self._event(
                    "response.reasoning_summary_part.added",
                    **place,
                    summary_index=0,
                    part=summary_text(""),
                ),
            ]
        added["arguments"] = ""
        return [self._event("response.output_item.added", output_index=index, item=added)]

    def _delta(self, event: BlockDelta) -> list[dict]:
        item = self.items.get(event.index)
        delta = event.delta
        if item is None:
            return []
        place = {"item_id": item.id, "output_index": item.output_index}
        if isinstance(delta, TextDelta) and isinstance(item.block, Text):
            item.text += delta.text
            return [
                self._event(
                    "response.output_text.delta",
                    **place,
                    content_index=0,
                    delta=delta.text,
                    logprobs=[],
                )
            ]
        if isinstance(delta, ThinkingDelta) and isinstance(item.block, Thinking):
            item.text += delta.thinking
            return [
                self._event(
                    "response.reasoning_summary_text.delta",
                    **place,
                    summary_index=0,
                    delta=delta.thinking,
                )
            ]
        if isinstance(delta, ToolInputDelta) and isinstance(item.block, ToolCall):
            item.text += delta.partial_json
            return [
                self._event(
                    "response.function_call_arguments.delta", **place, delta=delta.partial_json
                )
            ]
        # Signatures and opaque deltas have no Responses form.
        return []

    def _stop(self, event: BlockStop) -> list[dict]:
        item = self.items.get(event.index)
        if item is None:
            return []
        place = {"item_id": item.id, "output_index": item.output_index}
        events = []
        if isinstance(item.block, ToolCall) and not item.text:
            # A call with no arguments streamed still has an object of them.
            events += self._delta(
                BlockDelta(index=event.index, delta=ToolInputDelta(partial_json="{}"))
            )
        done = output_item(item.final(), item.id, "completed")
        if isinstance(item.block, Text):
            part = done["content"][0]
            events += [
                self._event(
                    "response.output_text.done",
                    **place,
                    content_index=0,
                    text=item.text,
                    logprobs=[],
                ),
                self._event("response.content_part.done", **place, content_index=0, part=part),
            ]
        elif isinstance(item.block, Thinking):
            events += [
                self._event(
                    "response.reasoning_summary_text.done", **place, summary_index=0, text=item.text
                ),
                self._event(
                    "response.reasoning_summary_part.done",
                    **place,
                    summary_index=0,
                    part=done["summary"][0],
                ),
            ]
        else:
            events.append(
                self._event("response.function_call_arguments.done", **place, arguments=item.text)
            )
        self.done[item.output_index] = done
        events.append(
            self._event("response.output_item.done", output_index=item.output_index, item=done)
        )
        return events
