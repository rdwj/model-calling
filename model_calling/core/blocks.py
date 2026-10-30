"""
Content blocks: the unit every turn is built from.

A turn is a list of blocks, not a string plus side channels. Tool calls,
tool results, and thinking are blocks in the turn where they occur, and
per-block metadata such as `cache_control` lives on the block it applies to.
"""

from typing import Annotated, Any, Literal

from pydantic import Field

from model_calling.core.base import CoreModel


class CacheControl(CoreModel):
    type: Literal["ephemeral"] = "ephemeral"
    ttl: Literal["5m", "1h"] | None = None


class Block(CoreModel):
    cache_control: CacheControl | None = None


class Text(Block):
    type: Literal["text"] = "text"
    text: str


class Base64Source(CoreModel):
    type: Literal["base64"] = "base64"
    media_type: str
    data: str


class UrlSource(CoreModel):
    type: Literal["url"] = "url"
    url: str


class FileSource(CoreModel):
    """A file previously uploaded to the provider, referenced by ID."""

    type: Literal["file"] = "file"
    file_id: str


ImageSource = Annotated[Base64Source | UrlSource | FileSource, Field(discriminator="type")]


class Image(Block):
    type: Literal["image"] = "image"
    source: ImageSource


class ToolCall(Block):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    input: dict[str, Any]
    # The arguments exactly as the wire sent them, when they arrived as text
    # that isn't a JSON object (small models often emit broken JSON). `input`
    # is then empty, and text-based encoders send this instead.
    raw_arguments: str | None = None


class Thinking(Block):
    """Model reasoning. `signature` is opaque and must be passed back verbatim."""

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None


class RedactedThinking(Block):
    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str


class Document(Block):
    """A document provided as context, containing structured source material.

    Supported by Anthropic and vLLM Messages backends; others refuse it.
    The source and other fields are kept as a dict since they're format-specific.
    """

    type: Literal["document"] = "document"
    data: dict[str, Any]


class Opaque(Block):
    """A block type the internal format doesn't model, kept verbatim.

    Only a translator for `format` can emit it; others must reject or drop it.
    """

    type: Literal["opaque"] = "opaque"
    format: str
    data: dict[str, Any]


ToolResultContent = Annotated[Text | Image | Opaque, Field(discriminator="type")]


class ToolResult(Block):
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    # None means the wire omitted content, which is distinct from an empty list.
    content: list[ToolResultContent] | None = None
    # Set when the wire sent content as a bare string rather than a block list.
    content_shorthand: bool = False
    is_error: bool | None = None


ContentBlock = Annotated[
    Text | Image | ToolCall | ToolResult | Thinking | RedactedThinking | Document | Opaque,
    Field(discriminator="type"),
]
