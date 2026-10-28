"""
Estimate a request's input tokens, for Anthropic's `count_tokens`.

No backend here has a native counter, and tokenizers differ by model, so
this is a deliberately simple estimate: about one token per four characters
of text and JSON, a flat cost per image, and a little overhead per message,
block, and tool. Treat it as a rough size, not a billing figure.
"""

import json
from typing import Any

from model_calling.core import (
    Image,
    Opaque,
    RedactedThinking,
    Request,
    Text,
    Thinking,
    Tool,
    ToolCall,
    ToolResult,
)

CHARS_PER_TOKEN = 4
# Role markers and turn structure.
TOKENS_PER_MESSAGE = 3
# Block boundaries.
TOKENS_PER_BLOCK = 2
# Tool definition structure, beyond its name, description, and schema.
TOKENS_PER_TOOL = 10
# A flat guess; the real cost depends on the image size and the model.
TOKENS_PER_IMAGE = 1000


def _text_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def _json_tokens(value: Any) -> int:
    return _text_tokens(json.dumps(value, separators=(",", ":")))


def _block_tokens(block: Any) -> int:
    tokens = TOKENS_PER_BLOCK
    if isinstance(block, Text):
        tokens += _text_tokens(block.text)
    elif isinstance(block, Image):
        tokens += TOKENS_PER_IMAGE
    elif isinstance(block, ToolCall):
        tokens += _text_tokens(block.name)
        tokens += _text_tokens(block.raw_arguments) or _json_tokens(block.input)
    elif isinstance(block, ToolResult):
        tokens += sum(_block_tokens(b) for b in block.content or [])
    elif isinstance(block, Thinking):
        tokens += _text_tokens(block.thinking)
    elif isinstance(block, RedactedThinking):
        tokens += _text_tokens(block.data)
    elif isinstance(block, Opaque):
        tokens += _json_tokens(block.data)
    return tokens


def _tool_tokens(tool: Tool | Opaque) -> int:
    if isinstance(tool, Opaque):
        return TOKENS_PER_TOOL + _json_tokens(tool.data)
    return (
        TOKENS_PER_TOOL
        + _text_tokens(tool.name)
        + _text_tokens(tool.description)
        + _json_tokens(tool.input_schema)
    )


def estimate_input_tokens(request: Request) -> int:
    """An estimate of `request`'s input tokens: system, messages, and tools."""
    total = sum(_block_tokens(b) for b in request.system or [])
    for message in request.messages:
        total += TOKENS_PER_MESSAGE + sum(_block_tokens(b) for b in message.content)
    total += sum(_tool_tokens(t) for t in request.tools or [])
    return total
