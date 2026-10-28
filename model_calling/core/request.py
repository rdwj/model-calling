"""
The internal request: what every ingress translates into and every backend
translates out of.

Chat Completions concepts (flat `content`, message-level `tool_calls`, the
`tool` and `developer` roles) are not modeled here; they are translation
concerns for that one format.
"""

from typing import Annotated, Any, Literal

from pydantic import Field

from model_calling.core.base import CoreModel
from model_calling.core.blocks import CacheControl, ContentBlock, Opaque, Text


class Message(CoreModel):
    """One turn. Leading system content belongs in `Request.system`; a
    `system` message is system content that arrives mid-conversation."""

    role: Literal["user", "assistant", "system"]
    content: list[ContentBlock]
    # Set when the wire sent content as a bare string rather than a block list.
    content_shorthand: bool = False


class Tool(CoreModel):
    """A client-side tool described by a JSON Schema."""

    type: Literal["tool"] = "tool"
    name: str
    description: str | None = None
    input_schema: dict[str, Any]
    cache_control: CacheControl | None = None


ToolSpec = Annotated[Tool | Opaque, Field(discriminator="type")]


class ToolChoice(CoreModel):
    mode: Literal["auto", "any", "tool", "none"]
    name: str | None = None
    parallel: bool | None = None


class ThinkingConfig(CoreModel):
    mode: Literal["enabled", "adaptive", "disabled"]
    budget_tokens: int | None = None
    # Provider-defined effort level such as "low" or "high".
    effort: str | None = None


class ResponseFormat(CoreModel):
    """Constrain output to JSON, optionally matching `json_schema`."""

    type: Literal["text", "json_object", "json_schema"]
    name: str | None = None
    json_schema: dict[str, Any] | None = None
    strict: bool | None = None


class Request(CoreModel):
    model: str
    messages: list[Message]
    system: list[Text] | None = None
    system_shorthand: bool = False
    tools: list[ToolSpec] | None = None
    tool_choice: ToolChoice | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop: list[str] | None = None
    thinking: ThinkingConfig | None = None
    response_format: ResponseFormat | None = None
    stream: bool | None = None
