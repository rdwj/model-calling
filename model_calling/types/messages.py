"""Anthropic Messages response types for the public messages() API."""

from typing import Annotated, Any, Literal

from pydantic import Discriminator, Tag, TypeAdapter

from model_calling.types.base import WireModel

# Tag for content blocks and deltas this module doesn't model. They parse
# into a fallback model that keeps every field instead of failing.
UNKNOWN = "unknown"


def _type_of(value: Any) -> Any:
    return value.get("type") if isinstance(value, dict) else getattr(value, "type", None)


def _known_or_unknown(*known: str) -> Discriminator:
    return Discriminator(lambda value: _type_of(value) if _type_of(value) in known else UNKNOWN)


class Usage(WireModel):
    """Token usage and billing information."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class TextBlock(WireModel):
    """A text content block."""

    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(WireModel):
    """Extended thinking content block with signature verification."""

    type: Literal["thinking"] = "thinking"
    thinking: str
    # Required, as in the SDK. vLLM's stream omits it at block start
    # (tests/fixtures/README.md); unsigned thinking from us carries "".
    signature: str


class RedactedThinkingBlock(WireModel):
    """Safety-redacted thinking block."""

    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str


class ToolUseBlock(WireModel):
    """Tool invocation by the model."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class UnknownContentBlock(WireModel):
    """Fallback for future content block types the SDK doesn't yet model."""

    type: str


ContentBlock = Annotated[
    Annotated[TextBlock, Tag("text")]
    | Annotated[ThinkingBlock, Tag("thinking")]
    | Annotated[RedactedThinkingBlock, Tag("redacted_thinking")]
    | Annotated[ToolUseBlock, Tag("tool_use")]
    | Annotated[UnknownContentBlock, Tag(UNKNOWN)],
    _known_or_unknown("text", "thinking", "redacted_thinking", "tool_use"),
]


class Message(WireModel):
    """Anthropic Messages API response."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock]
    model: str
    stop_reason: str | None = None  # Loose str for forward compatibility with new stop reasons
    stop_sequence: str | None = None
    usage: Usage


class TextDelta(WireModel):
    """Incremental text in a streaming delta."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class ThinkingDelta(WireModel):
    """Incremental thinking text in a streaming delta."""

    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str


class SignatureDelta(WireModel):
    """Thinking block signature delivered at block end."""

    type: Literal["signature_delta"] = "signature_delta"
    signature: str


class InputJSONDelta(WireModel):
    """Incremental tool input JSON in a streaming delta."""

    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class UnknownDelta(WireModel):
    """Fallback for future delta types."""

    type: str


Delta = Annotated[
    Annotated[TextDelta, Tag("text_delta")]
    | Annotated[ThinkingDelta, Tag("thinking_delta")]
    | Annotated[SignatureDelta, Tag("signature_delta")]
    | Annotated[InputJSONDelta, Tag("input_json_delta")]
    | Annotated[UnknownDelta, Tag(UNKNOWN)],
    _known_or_unknown("text_delta", "thinking_delta", "signature_delta", "input_json_delta"),
]


class MessageStartEvent(WireModel):
    """Stream event: message metadata available."""

    type: Literal["message_start"] = "message_start"
    message: Message


class ContentBlockStartEvent(WireModel):
    """Stream event: new content block starting."""

    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: ContentBlock


class ContentBlockDeltaEvent(WireModel):
    """Stream event: incremental content within a block."""

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: Delta


class ContentBlockStopEvent(WireModel):
    """Stream event: content block complete."""

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaUsage(WireModel):
    """Cumulative usage in message_delta event."""

    output_tokens: int
    input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class MessageDelta(WireModel):
    """Message-level changes in message_delta event."""

    stop_reason: str | None = None
    stop_sequence: str | None = None


class MessageDeltaEvent(WireModel):
    """Stream event: message-level delta (stop reason, final usage)."""

    type: Literal["message_delta"] = "message_delta"
    delta: MessageDelta
    usage: MessageDeltaUsage


class MessageStopEvent(WireModel):
    """Stream event: message complete."""

    type: Literal["message_stop"] = "message_stop"


MessageStreamEvent = Annotated[
    Annotated[MessageStartEvent, Tag("message_start")]
    | Annotated[ContentBlockStartEvent, Tag("content_block_start")]
    | Annotated[ContentBlockDeltaEvent, Tag("content_block_delta")]
    | Annotated[ContentBlockStopEvent, Tag("content_block_stop")]
    | Annotated[MessageDeltaEvent, Tag("message_delta")]
    | Annotated[MessageStopEvent, Tag("message_stop")],
    Discriminator(_type_of),
]

_event_adapter = TypeAdapter(MessageStreamEvent)


def parse_stream_event(data: dict) -> MessageStreamEvent:
    """Parse a Messages stream event dict into its typed model."""
    return _event_adapter.validate_python(data)
