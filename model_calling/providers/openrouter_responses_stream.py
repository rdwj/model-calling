"""
An OpenAI Responses SSE stream, line by line, to internal events.

Responses streams send:
- response.created (the initial response object, status in_progress)
- response.in_progress
- response.output_item.added for each output item
- item-specific events (text deltas, function call deltas, reasoning deltas)
- response.output_item.done for each completed item
- response.completed or response.incomplete (final response with all output)

All events have a sequence_number. [DONE] is not used in Responses streams.
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
from model_calling.formats.responses import INCOMPLETE_REASONS
from model_calling.providers.base import BAD_GATEWAY, StreamParser, new_id

DATA_PREFIX = "data:"


class ResponsesStreamParser(StreamParser):
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self.started = False
        # Output item id -> internal block index.
        self.items: dict[str, int] = {}
        self.index = -1
        # The open block's item id, if any.
        self.open_item: str | None = None
        self.stop_reason: str | None = None
        self.usage: Usage | None = None
        self.done = False

    def feed(self, line: str) -> list[StreamEvent]:
        if not line.startswith(DATA_PREFIX) or self.done:
            return []
        data = line[len(DATA_PREFIX) :].strip()
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            return [self._error(f"{self.provider} sent a non-JSON chunk: {data!r}")]
        if not isinstance(chunk, dict):
            return [self._error(f"{self.provider} sent a chunk that isn't an object: {data!r}")]

        event_type = chunk.get("type")
        if event_type == "error":
            message = chunk.get("message", "unknown error")
            kind = chunk.get("code")
            return [self._error(f"{self.provider} error: {message}", kind)]

        if event_type == "response.created":
            return self._created(chunk.get("response") or {})
        if event_type == "response.output_item.added":
            return self._item_added(chunk)
        # OpenRouter sends type-specific delta events like response.output_text.delta
        # instead of response.output_item.delta
        if event_type == "response.output_text.delta":
            return self._text_delta(chunk)
        if event_type in ("response.function_call.delta", "response.function_call_arguments.delta"):
            return self._function_delta(chunk)
        if event_type == "response.reasoning_text.delta":
            return self._reasoning_delta(chunk)
        # Generic delta format (if OpenAI uses it)
        if event_type == "response.output_item.delta":
            return self._item_delta(chunk)
        if event_type == "response.output_item.done":
            return self._item_done(chunk)
        if event_type in ("response.completed", "response.incomplete"):
            return self._finished(chunk, event_type == "response.incomplete")

        # Ignore other event types (response.in_progress, response.content_part.*, etc.).
        return []

    def close(self) -> list[StreamEvent]:
        if self.done:
            return []
        if self.stop_reason is not None:
            # Finished, but never got the final event.
            return self._close_open() + [
                MessageDelta(stop_reason=self.stop_reason, usage=self.usage),
                MessageStop(),
            ]
        return [self._error(f"{self.provider} closed the stream before it finished")]

    # --- internals --------------------------------------------------------

    def _created(self, response: dict) -> list[StreamEvent]:
        if self.started:
            return []
        self.started = True
        usage_dict = response.get("usage") or {}
        usage = (
            Usage(
                input_tokens=usage_dict.get("input_tokens"),
                output_tokens=usage_dict.get("output_tokens"),
                cache_read_tokens=(usage_dict.get("input_tokens_details") or {}).get(
                    "cached_tokens"
                ),
                cache_write_tokens=(usage_dict.get("input_tokens_details") or {}).get(
                    "cache_write_tokens"
                ),
                reasoning_tokens=(usage_dict.get("output_tokens_details") or {}).get(
                    "reasoning_tokens"
                ),
            )
            if usage_dict
            else None
        )
        self.usage = usage
        return [
            MessageStart(
                id=response.get("id") or new_id("resp-"),
                model=response.get("model") or self.model,
                usage=usage,
            )
        ]

    def _item_added(self, chunk: dict) -> list[StreamEvent]:
        item = chunk.get("item") or {}
        item_id = item.get("id")
        if not item_id:
            return [self._error(f"{self.provider} sent output_item.added without an id")]

        item_type = item.get("type")
        self.index += 1
        self.items[item_id] = self.index
        self.open_item = item_id

        if item_type == "message":
            block = Text(text="")
        elif item_type == "function_call":
            block = ToolCall(
                id=item.get("call_id", new_id("fc-")),
                name=item.get("name") or "",
                input={},
            )
        elif item_type == "reasoning":
            block = Thinking(thinking="")
        else:
            return [self._error(f"{self.provider} sent unknown item type {item_type!r}")]

        return [BlockStart(index=self.index, block=block)]

    def _text_delta(self, chunk: dict) -> list[StreamEvent]:
        """Handle response.output_text.delta events (OpenRouter format)."""
        item_id = chunk.get("item_id")
        if not item_id:
            return []
        index = self.items.get(item_id)
        if index is None:
            return [self._error(f"{self.provider} sent delta for unknown item {item_id}")]
        text = chunk.get("delta", "")
        return [BlockDelta(index=index, delta=TextDelta(text=text))]

    def _function_delta(self, chunk: dict) -> list[StreamEvent]:
        """Handle response.function_call.delta events (OpenRouter format)."""
        item_id = chunk.get("item_id")
        if not item_id:
            return []
        index = self.items.get(item_id)
        if index is None:
            return [self._error(f"{self.provider} sent delta for unknown item {item_id}")]
        arguments = chunk.get("delta", "")
        return [BlockDelta(index=index, delta=ToolInputDelta(partial_json=arguments))]

    def _reasoning_delta(self, chunk: dict) -> list[StreamEvent]:
        """Handle response.reasoning_text.delta events (OpenRouter format)."""
        item_id = chunk.get("item_id")
        if not item_id:
            return []
        index = self.items.get(item_id)
        if index is None:
            return [self._error(f"{self.provider} sent delta for unknown item {item_id}")]
        text = chunk.get("delta", "")
        return [BlockDelta(index=index, delta=ThinkingDelta(thinking=text))]

    def _item_delta(self, chunk: dict) -> list[StreamEvent]:
        """Handle response.output_item.delta events (generic format)."""
        item_id = chunk.get("item_id")
        if not item_id:
            return []
        index = self.items.get(item_id)
        if index is None:
            return [self._error(f"{self.provider} sent delta for unknown item {item_id}")]

        delta_data = chunk.get("delta") or {}
        delta_type = delta_data.get("type")

        if delta_type == "output_text":
            text = delta_data.get("text", "")
            return [BlockDelta(index=index, delta=TextDelta(text=text))]
        elif delta_type == "arguments":
            arguments = delta_data.get("arguments", "")
            return [BlockDelta(index=index, delta=ToolInputDelta(partial_json=arguments))]
        elif delta_type in ("summary_text", "reasoning_text"):
            # OpenRouter uses reasoning_text, but also support summary_text for compatibility.
            text = delta_data.get("text", "")
            return [BlockDelta(index=index, delta=ThinkingDelta(thinking=text))]

        return []

    def _item_done(self, chunk: dict) -> list[StreamEvent]:
        item_id = chunk.get("item_id")
        if not item_id or item_id not in self.items:
            return []
        if self.open_item == item_id:
            self.open_item = None
            return [BlockStop(index=self.items[item_id])]
        return []

    def _finished(self, chunk: dict, incomplete: bool) -> list[StreamEvent]:
        if not self.started:
            return [self._error(f"{self.provider} ended the stream without a response")]
        self.done = True

        response = chunk.get("response") or {}
        usage_dict = response.get("usage") or {}
        if usage_dict:
            self.usage = Usage(
                input_tokens=usage_dict.get("input_tokens"),
                output_tokens=usage_dict.get("output_tokens"),
                cache_read_tokens=(usage_dict.get("input_tokens_details") or {}).get(
                    "cached_tokens"
                ),
                cache_write_tokens=(usage_dict.get("input_tokens_details") or {}).get(
                    "cache_write_tokens"
                ),
                reasoning_tokens=(usage_dict.get("output_tokens_details") or {}).get(
                    "reasoning_tokens"
                ),
            )

        if incomplete:
            incomplete_details = response.get("incomplete_details") or {}
            reason = incomplete_details.get("reason")
            # Map Responses incomplete reason back to internal stop_reason.
            for internal, external in INCOMPLETE_REASONS.items():
                if external == reason:
                    self.stop_reason = internal
                    break
            if self.stop_reason is None:
                self.stop_reason = "max_tokens"
        else:
            # Completed - check output to determine stop reason.
            output = response.get("output") or []
            if output:
                last_item = output[-1]
                if last_item.get("type") == "function_call":
                    self.stop_reason = "tool_use"
                else:
                    self.stop_reason = "end_turn"
            else:
                self.stop_reason = "end_turn"

        return self._close_open() + [
            MessageDelta(stop_reason=self.stop_reason, usage=self.usage),
            MessageStop(),
        ]

    def _close_open(self) -> list[StreamEvent]:
        if self.open_item is not None:
            index = self.items[self.open_item]
            self.open_item = None
            return [BlockStop(index=index)]
        return []

    @staticmethod
    def _error(message: str, error_type: str | None = None) -> StreamError:
        return StreamError(status=BAD_GATEWAY, message=message, error_type=error_type)
