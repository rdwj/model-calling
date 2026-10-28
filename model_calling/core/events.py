"""
Internal stream events.

A stream is `MessageStart`, then for each block a `BlockStart`, any number of
`BlockDelta`s, and a `BlockStop`, then `MessageDelta` and `MessageStop`.
Block indexes count from 0 in order of first appearance. The block in
`BlockStart` carries empty content (`""` text, `{}` tool input); deltas fill
it in. Usage on `MessageDelta` is cumulative for the whole response.
"""

from typing import Annotated, Any, Literal

from pydantic import Field

from model_calling.core.base import CoreModel
from model_calling.core.blocks import ContentBlock
from model_calling.core.response import StopReason, Usage


class TextDelta(CoreModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingDelta(CoreModel):
    type: Literal["thinking"] = "thinking"
    thinking: str


class SignatureDelta(CoreModel):
    type: Literal["signature"] = "signature"
    signature: str


class ToolInputDelta(CoreModel):
    """A fragment of the tool call's input as JSON text, not yet parseable alone."""

    type: Literal["tool_input"] = "tool_input"
    partial_json: str


class OpaqueDelta(CoreModel):
    """A delta type the internal format doesn't model, kept verbatim for `format`."""

    type: Literal["opaque"] = "opaque"
    format: str
    data: dict[str, Any]


Delta = Annotated[
    TextDelta | ThinkingDelta | SignatureDelta | ToolInputDelta | OpaqueDelta,
    Field(discriminator="type"),
]


class MessageStart(CoreModel):
    type: Literal["message_start"] = "message_start"
    id: str
    model: str
    usage: Usage | None = None


class BlockStart(CoreModel):
    type: Literal["block_start"] = "block_start"
    index: int
    block: ContentBlock


class BlockDelta(CoreModel):
    type: Literal["block_delta"] = "block_delta"
    index: int
    delta: Delta


class BlockStop(CoreModel):
    type: Literal["block_stop"] = "block_stop"
    index: int


class MessageDelta(CoreModel):
    type: Literal["message_delta"] = "message_delta"
    stop_reason: StopReason | None = None
    stop_sequence: str | None = None
    usage: Usage | None = None


class MessageStop(CoreModel):
    type: Literal["message_stop"] = "message_stop"


class StreamError(CoreModel):
    """The backend failed mid-stream. `status` is the HTTP status it maps to;
    `error_type` is the backend's own error kind, such as `overloaded_error`."""

    type: Literal["error"] = "error"
    status: int
    message: str
    error_type: str | None = None


StreamEvent = Annotated[
    MessageStart | BlockStart | BlockDelta | BlockStop | MessageDelta | MessageStop | StreamError,
    Field(discriminator="type"),
]
