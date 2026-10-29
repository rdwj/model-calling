"""
OpenRouter backend for OpenAI Responses at `{base_url}/responses`.

Stateless: store is accepted and reported false; requests needing stored state
were already refused by the ingress.
"""

import json
import logging
from typing import Any, ClassVar

from model_calling.core import (
    Base64Source,
    FileSource,
    Image,
    Message,
    Opaque,
    RedactedThinking,
    Request,
    Response,
    Text,
    Thinking,
    Tool,
    ToolCall,
    ToolResult,
)
from model_calling.formats import TranslationError
from model_calling.formats.responses import (
    FORMAT,
    INCOMPLETE_REASONS,
    arguments,
    item_id,
    output_text,
    summary_text,
)
from model_calling.errors import APIError
from model_calling.providers.base import Provider, StreamParser, new_id
from model_calling.providers.openrouter_responses_stream import ResponsesStreamParser

logger = logging.getLogger(__name__)

RESPONSES_PATH = "/responses"
_MANAGED_FIELDS = ("stream",)
_TOOL_CHOICE_MODES = {"auto": "auto", "none": "none", "any": "required"}


def _text(blocks: list) -> str:
    return "".join(b.text for b in blocks if isinstance(b, Text))


def _check_part(block: Any) -> None:
    if isinstance(block, Opaque) and block.format != FORMAT:
        raise TranslationError(
            f"openrouter_responses cannot accept a {block.format} {block.data.get('type')!r} block"
        )
    if isinstance(block, Image) and isinstance(block.source, FileSource):
        raise TranslationError(
            "openrouter_responses cannot accept an image by file id (use image_url)"
        )


def _part_out(block: Any) -> dict:
    if isinstance(block, Text):
        return {"type": "input_text", "text": block.text}
    if isinstance(block, Image):
        source = block.source
        if isinstance(source, Base64Source):
            url = f"data:{source.media_type};base64,{source.data}"
        else:
            url = source.url
        return {
            "type": "input_image",
            "image_url": url,
            **block.extras_for(FORMAT),
        }
    return block.data


def _content_out(blocks: list) -> str | list:
    """A string when content is plain text, otherwise content parts."""
    if all(isinstance(b, Text) for b in blocks) and len(blocks) == 1:
        return blocks[0].text
    return [_part_out(b) for b in blocks]


def _tool_call_out(block: ToolCall | Opaque) -> dict:
    if isinstance(block, Opaque):
        return block.data
    return {
        "type": "function_call",
        "call_id": block.id,
        "name": block.name,
        "arguments": arguments(block),
    }


def _reasoning_out(blocks: list) -> list[dict]:
    """Reasoning items from thinking blocks."""
    items = []
    for block in blocks:
        if isinstance(block, Thinking):
            items.append({"type": "reasoning", "summary": [summary_text(block.thinking)]})
    return items


class OpenRouterResponsesProvider(Provider):
    name = "openrouter_responses"

    def endpoint(self) -> str:
        return f"{self.base_url}{RESPONSES_PATH}"

    # --- request ----------------------------------------------------------

    def unsupported_params(self, request: Request) -> list[str]:
        names = []
        thinking = request.thinking
        if thinking is not None and thinking.budget_tokens is not None and thinking.effort is None:
            # Responses takes reasoning.effort; a token budget has no equivalent.
            names.append("thinking.budget_tokens")
        return names

    def build_request(self, request: Request) -> dict:
        input_items: list[dict] = []

        # System prompt becomes instructions.
        instructions = None
        if request.system:
            instructions = _text(request.system)

        # Convert messages to input items.
        for msg in request.messages:
            for block in msg.content:
                _check_part(block)

            if msg.role == "system":
                # Inline system messages become message items.
                content = _content_out([b for b in msg.content if isinstance(b, Text)])
                input_items.append(
                    {
                        "type": "message",
                        "role": "system",
                        "content": content,
                    }
                )
            elif msg.role == "user":
                # Tool results first, then the rest as a user message.
                results = [b for b in msg.content if isinstance(b, ToolResult)]
                for result in results:
                    content_blocks = result.content or []
                    if not all(isinstance(b, Text) for b in content_blocks):
                        raise TranslationError(
                            "openrouter_responses tool results may contain only text"
                        )
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": result.tool_call_id,
                            "output": _text(content_blocks),
                        }
                    )

                rest = [b for b in msg.content if not isinstance(b, ToolResult)]
                if rest:
                    input_items.append(
                        {
                            "type": "message",
                            "role": "user",
                            "content": _content_out(rest),
                        }
                    )
            elif msg.role == "assistant":
                # Reasoning, then tool calls, then text.
                input_items.extend(_reasoning_out(msg.content))

                for block in msg.content:
                    if isinstance(block, ToolCall):
                        input_items.append(_tool_call_out(block))
                    elif isinstance(block, Text):
                        input_items.append(
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": block.text,
                            }
                        )
                    elif isinstance(block, Opaque) and "id" in block.data:
                        input_items.append(block.data)

        body: dict[str, Any] = {
            k: v for k, v in request.extras_for(FORMAT).items() if k not in _MANAGED_FIELDS
        }
        body.update(model=self.model, input=input_items)

        if instructions is not None:
            body["instructions"] = instructions

        if request.tools:
            body["tools"] = [self._tool_out(t) for t in request.tools]

        if request.tool_choice is not None:
            body.update(self._tool_choice_out(request))

        params = {
            "max_output_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "reasoning": self._reasoning_out(request),
            "text": self._text_format_out(request),
        }
        body.update({k: v for k, v in params.items() if v is not None})

        # Store is always false (stateless).
        body["store"] = False

        return body

    def _tool_out(self, tool: Tool | Opaque) -> dict:
        if isinstance(tool, Opaque):
            if tool.format != FORMAT:
                raise TranslationError(
                    f"openrouter_responses cannot run {tool.format} tool {tool.data.get('type')!r}"
                )
            return tool.data
        out: dict[str, Any] = {
            "type": "function",
            "name": tool.name,
        }
        if tool.description is not None:
            out["description"] = tool.description
        out["parameters"] = tool.input_schema
        out.update(tool.extras_for(FORMAT))
        if tool.strict is not None:
            out["strict"] = tool.strict
        return out

    @staticmethod
    def _tool_choice_out(request: Request) -> dict:
        choice = request.tool_choice
        out: dict[str, Any] = {}
        if choice.mode == "tool":
            out["tool_choice"] = {"type": "function", "name": choice.name}
        elif choice.mode != "auto" or request.tools:
            out["tool_choice"] = _TOOL_CHOICE_MODES[choice.mode]
        if choice.parallel is not None:
            out["parallel_tool_calls"] = choice.parallel
        return out

    def _reasoning_out(self, request: Request) -> dict | None:
        thinking = request.thinking
        if thinking is None:
            return None
        if thinking.mode == "disabled":
            return {"effort": "none"}
        if thinking.effort is None:
            logger.debug(
                "openrouter_responses takes reasoning effort, not a token budget; not sent"
            )
            return None
        return {"effort": thinking.effort}

    @staticmethod
    def _text_format_out(request: Request) -> dict | None:
        fmt = request.response_format
        if fmt is None:
            return None
        if fmt.type != "json_schema":
            return {"format": {"type": fmt.type}}
        schema: dict[str, Any] = {
            "type": "json_schema",
            "name": fmt.name or "response",
        }
        if fmt.json_schema is not None:
            schema["schema"] = fmt.json_schema
        if fmt.strict is not None:
            schema["strict"] = fmt.strict
        return {"format": schema}

    # --- response ---------------------------------------------------------

    def parse_response(self, body: dict) -> Response:
        if body.get("error"):
            raise APIError(
                f"openrouter_responses returned an error: {self.error_message(body)}",
                provider=self.name,
                body=body,
            )

        output = body.get("output") or []
        blocks: list = []

        for item in output:
            item_type = item.get("type")
            if item_type == "message":
                # Text content.
                content = item.get("content") or []
                for part in content:
                    if part.get("type") == "output_text":
                        blocks.append(Text(text=part["text"]))
            elif item_type == "function_call":
                # Tool call.
                call = ToolCall(
                    id=item.get("call_id", new_id("fc-")),
                    name=item["name"],
                    input=json.loads(item["arguments"]) if item.get("arguments") else {},
                    raw_arguments=item.get("arguments"),
                )
                blocks.append(call)
            elif item_type == "reasoning":
                # Thinking. OpenRouter uses content with reasoning_text, not summary with summary_text.
                content = item.get("content") or []
                for part in content:
                    if part.get("type") == "reasoning_text":
                        blocks.append(Thinking(thinking=part["text"]))
                # Also check summary for compatibility.
                summary = item.get("summary") or []
                for part in summary:
                    if part.get("type") == "summary_text":
                        blocks.append(Thinking(thinking=part["text"]))

        # Map status to stop_reason.
        status = body.get("status")
        incomplete_details = body.get("incomplete_details")
        stop_reason = None

        if status == "incomplete" and incomplete_details:
            reason = incomplete_details.get("reason")
            # Reverse lookup from INCOMPLETE_REASONS.
            for internal, external in INCOMPLETE_REASONS.items():
                if external == reason:
                    stop_reason = internal
                    break
            if stop_reason is None:
                stop_reason = "max_tokens"
        elif status == "completed":
            # Check if it ended with tool calls.
            if blocks and isinstance(blocks[-1], ToolCall):
                stop_reason = "tool_use"
            else:
                stop_reason = "end_turn"

        usage_dict = body.get("usage") or {}
        from model_calling.core import Usage

        usage = Usage(
            input_tokens=usage_dict.get("input_tokens"),
            output_tokens=usage_dict.get("output_tokens"),
            cache_read_tokens=(usage_dict.get("input_tokens_details") or {}).get("cached_tokens"),
            cache_write_tokens=(usage_dict.get("input_tokens_details") or {}).get(
                "cache_write_tokens"
            ),
            reasoning_tokens=(usage_dict.get("output_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
        )

        return Response(
            id=body.get("id") or new_id("resp-"),
            model=body.get("model") or self.model,
            content=blocks,
            stop_reason=stop_reason,
            usage=usage,
        )

    def stream_parser(self) -> StreamParser:
        return ResponsesStreamParser(self.name, self.model)
