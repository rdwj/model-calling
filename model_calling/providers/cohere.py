"""
Cohere's v2 API: `/v2/chat` (https://docs.cohere.com/reference/chat) and
`/v2/embed` (https://docs.cohere.com/reference/embed).

`base_url` is the API root including the version
(`https://api.cohere.com/v2`), as for the other hosted backends. The system
prompt is a `system` message, not a top-level field. Tool call arguments
travel as JSON strings. Thinking is `{type: enabled|disabled, token_budget}`
with no effort levels, and reasoning comes back as `thinking` content
blocks. The model may write a `tool_plan` before its tool calls; it becomes
a text block.

Where v2 can't express a request, the provider refuses rather than
approximates: a named tool_choice (v2 has only `REQUIRED` and `NONE`),
`parallel_tool_calls: false`, strict schemas on some tools but not others
(`strict_tools` covers every tool), an effort level without a token budget,
`response_format` together with tools, and more than five stop sequences.
"""

import json
import logging
from typing import Any

from model_calling.core import (
    Base64Source,
    Document,
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
from model_calling.core.embeddings import EmbeddingRequest, EmbeddingResponse
from model_calling.errors import APIError
from model_calling.formats import TranslationError
from model_calling.formats.embeddings import FORMAT as EMBEDDINGS_FORMAT
from model_calling.providers.base import Provider, StreamParser, new_id
from model_calling.providers.cohere_stream import (
    FAILED_FINISHES,
    STOP_REASONS,
    CohereStreamParser,
    failure_message,
    tool_call_in,
    usage_in,
)

logger = logging.getLogger(__name__)

CHAT_PATH = "/chat"
EMBED_PATH = "/embed"
# `stop_sequences` takes "a list of up to 5 strings".
MAX_STOP_SEQUENCES = 5
_TOOL_CHOICES = {"any": "REQUIRED", "none": "NONE"}
# `input_type` is required for embed v3 and newer. A client may name one as
# an extra field of its embeddings request; documents are the default.
INPUT_TYPE = "input_type"
DEFAULT_INPUT_TYPE = "search_document"
FLOAT = "float"


def _strict_tools(request: Request) -> bool | None:
    """True when every tool asks for strict schemas, None when none does,
    False when only some do (which `strict_tools` can't say)."""
    flags = {isinstance(t, Tool) and t.strict is True for t in request.tools or ()}
    if flags == {True}:
        return True
    return None if flags <= {False} else False


def _unspoken_effort(request: Request) -> bool:
    thinking = request.thinking
    return (
        thinking is not None
        and thinking.mode != "disabled"
        and thinking.effort is not None
        and thinking.budget_tokens is None
    )


def _format_with_tools(request: Request) -> bool:
    """v2's response_format "is not supported when used in combinations with
    the documents or tools parameters"."""
    fmt = request.response_format
    return fmt is not None and fmt.type != "text" and bool(request.tools)


def _text_parts(blocks: list) -> list[dict]:
    return [{"type": "text", "text": b.text} for b in blocks]


def _text_content(blocks: list, shorthand: bool) -> str | list[dict]:
    """A string when the wire sent one, otherwise text blocks."""
    if shorthand and len(blocks) == 1:
        return blocks[0].text
    return _text_parts(blocks)


class CohereProvider(Provider):
    name = "cohere"

    def endpoint(self) -> str:
        return f"{self.base_url}{CHAT_PATH}"

    # --- request ----------------------------------------------------------

    def unsupported_params(self, request: Request) -> list[str]:
        names = []
        choice = request.tool_choice
        sends_tools = bool(request.tools) and not (choice and choice.mode == "none")
        if choice is not None and choice.mode == "tool":
            names.append("tool_choice")
        if sends_tools and choice is not None and choice.parallel is False:
            names.append("parallel_tool_calls")
        if _strict_tools(request) is False:
            names.append("tools.strict")
        if _unspoken_effort(request):
            names.append("thinking.effort")
        if _format_with_tools(request):
            names.append("response_format")
        return names

    def build_request(self, request: Request) -> dict:
        messages: list[dict] = []
        if request.system:
            # Chat Completions has no system shorthand to keep, so one block is
            # a string whatever the ingress, and several stay apart.
            content = _text_content(request.system, shorthand=True)
            messages.append({"role": "system", "content": content})
        for msg in request.messages:
            messages.extend(self._message_out(msg))
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if request.tools:
            body["tools"] = [self._tool_out(t) for t in request.tools]
            if _strict_tools(request):
                body["strict_tools"] = True
            body.update(self._tool_choice_out(request))
        if request.stop is not None and len(request.stop) > MAX_STOP_SEQUENCES:
            raise TranslationError(
                f"Cohere takes at most {MAX_STOP_SEQUENCES} stop sequences, "
                f"not {len(request.stop)}"
            )
        params = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "p": request.top_p,
            "k": request.top_k,
            "seed": request.seed,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "stop_sequences": request.stop,
            "thinking": self._thinking_out(request),
            "response_format": self._response_format_out(request),
        }
        body.update({k: v for k, v in params.items() if v is not None})
        return body

    def _message_out(self, msg: Message) -> list[dict]:
        """One internal turn as Cohere messages: a `tool` message per tool
        result, then the rest of the turn."""
        for block in msg.content:
            if isinstance(block, Document):
                raise TranslationError("Cohere cannot accept document blocks")
            if isinstance(block, Opaque):
                raise TranslationError(
                    f"Cohere cannot accept a {block.format} {block.data.get('type')!r} block"
                )
            if isinstance(block, Image) and isinstance(block.source, FileSource):
                raise TranslationError("Cohere cannot accept an image by file id")
        out = [self._tool_result_out(b) for b in msg.content if isinstance(b, ToolResult)]
        rest = [b for b in msg.content if not isinstance(b, ToolResult)]
        if msg.role == "assistant":
            return out + [self._assistant_out(msg)]
        if out and not rest:
            return out
        if msg.role == "system":
            if not all(isinstance(b, Text) for b in rest):
                raise TranslationError("a system message may contain only text")
            return out + [{"role": "system", "content": _text_content(rest, msg.content_shorthand)}]
        if not all(isinstance(b, (Text, Image)) for b in rest):
            raise TranslationError("a Cohere user message may contain only text and images")
        if all(isinstance(b, Text) for b in rest):
            content: str | list[dict] = _text_content(rest, msg.content_shorthand)
        else:
            content = [self._part_out(b) for b in rest]
        return out + [{"role": "user", "content": content}]

    @staticmethod
    def _part_out(block: Text | Image) -> dict:
        if isinstance(block, Text):
            return {"type": "text", "text": block.text}
        source = block.source
        if isinstance(source, Base64Source):
            url = f"data:{source.media_type};base64,{source.data}"
        else:
            url = source.url
        return {"type": "image_url", "image_url": {"url": url}}

    def _assistant_out(self, msg: Message) -> dict:
        content, calls = [], []
        for b in msg.content:
            if isinstance(b, Text):
                content.append({"type": "text", "text": b.text})
            elif isinstance(b, Thinking):
                # v2 takes reasoning back as thinking blocks; it has no signature.
                content.append({"type": "thinking", "thinking": b.thinking})
            elif isinstance(b, ToolCall):
                calls.append(self._tool_call_out(b))
            elif isinstance(b, RedactedThinking):
                logger.debug("Cohere takes no redacted thinking; dropping it")
            elif isinstance(b, Image):
                raise TranslationError("a Cohere assistant message cannot contain images")
        out: dict[str, Any] = {"role": "assistant"}
        if msg.content_shorthand and len(content) == 1 and content[0]["type"] == "text":
            out["content"] = content[0]["text"]
        elif content or not calls:
            out["content"] = content or ""
        if calls:
            out["tool_calls"] = calls
        return out

    @staticmethod
    def _tool_call_out(call: ToolCall) -> dict:
        arguments = call.raw_arguments if call.raw_arguments is not None else json.dumps(call.input)
        return {
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": arguments},
        }

    @staticmethod
    def _tool_result_out(result: ToolResult) -> dict:
        # v2 has no error flag on tool results; the content says what failed.
        blocks = result.content or []
        if not all(isinstance(b, Text) for b in blocks):
            raise TranslationError("Cohere tool results may contain only text")
        content = _text_content(blocks, result.content_shorthand) if blocks else ""
        return {"role": "tool", "tool_call_id": result.tool_call_id, "content": content}

    @staticmethod
    def _tool_out(tool: Tool | Opaque) -> dict:
        if not isinstance(tool, Tool):
            raise TranslationError(
                f"Cohere cannot run {tool.format} tool {tool.data.get('type')!r}"
            )
        fn: dict[str, Any] = {"name": tool.name, "parameters": tool.input_schema}
        if tool.description is not None:
            fn["description"] = tool.description
        return {"type": "function", "function": fn}

    @staticmethod
    def _tool_choice_out(request: Request) -> dict:
        choice = request.tool_choice
        if choice is None or choice.mode == "auto":
            return {}
        if choice.mode == "tool":
            logger.warning("Cohere cannot force a named tool; ignoring tool_choice")
            return {}
        return {"tool_choice": _TOOL_CHOICES[choice.mode]}

    @staticmethod
    def _thinking_out(request: Request) -> dict | None:
        thinking = request.thinking
        if thinking is None:
            return None
        if thinking.mode == "disabled":
            return {"type": "disabled"}
        # Adaptive thinking is v2's default: enabled, with no budget.
        out: dict[str, Any] = {"type": "enabled"}
        if thinking.budget_tokens is not None:
            out["token_budget"] = thinking.budget_tokens
        elif thinking.effort is not None:
            logger.debug("Cohere takes a thinking token budget, not an effort; not sent")
        return out

    @staticmethod
    def _response_format_out(request: Request) -> dict | None:
        fmt = request.response_format
        if fmt is None:
            return None
        if _format_with_tools(request):
            logger.warning("Cohere cannot constrain output while tools are offered; ignoring it")
            return None
        if fmt.type == "text":
            return {"type": "text"}
        # v2 has no schema name, and always holds output to a given schema.
        out: dict[str, Any] = {"type": "json_object"}
        if fmt.json_schema is not None:
            out["json_schema"] = fmt.json_schema
        return out

    # --- response ---------------------------------------------------------

    def parse_response(self, body: dict) -> Response:
        finish = body.get("finish_reason")
        if finish in FAILED_FINISHES:
            raise APIError(failure_message(finish, None), provider=self.name, body=body)
        msg = body.get("message")
        if not isinstance(msg, dict):
            raise TranslationError("response has no message")
        blocks: list = []
        for item in msg.get("content") or []:
            if item.get("type") == "thinking":
                blocks.append(Thinking(thinking=item["thinking"]))
            elif item.get("type") == "text":
                blocks.append(Text(text=item["text"]))
        if msg.get("tool_plan"):
            blocks.append(Text(text=msg["tool_plan"]))
        blocks.extend(tool_call_in(tc) for tc in msg.get("tool_calls") or [])
        return Response(
            id=body.get("id") or new_id(),
            model=self.model,
            content=blocks,
            stop_reason=STOP_REASONS.get(finish),
            usage=usage_in(body.get("usage")),
        )

    def stream_parser(self) -> StreamParser:
        return CohereStreamParser(self.model)

    # --- embeddings -------------------------------------------------------

    def embedding_endpoint(self) -> str:
        return f"{self.base_url}{EMBED_PATH}"

    def unsupported_embedding_params(self, request: EmbeddingRequest) -> list[str]:
        # `user` only labels the caller, so it is dropped silently, as for chat.
        return sorted(k for k in request.extras_for(EMBEDDINGS_FORMAT) if k != INPUT_TYPE)

    def build_embedding_request(self, request: EmbeddingRequest) -> dict:
        if not all(isinstance(x, str) for x in request.input):
            raise TranslationError("Cohere embeds text only, not token arrays")
        input_type = request.extras_for(EMBEDDINGS_FORMAT).get(INPUT_TYPE, DEFAULT_INPUT_TYPE)
        body: dict[str, Any] = {
            "model": self.model,
            "texts": request.input,
            "input_type": input_type,
            # Floats always; formats.embeddings applies a client's base64.
            "embedding_types": [FLOAT],
        }
        if request.dimensions is not None:
            # Only embed-v4 and newer take it; older models refuse it.
            body["output_dimension"] = request.dimensions
        return body

    def parse_embedding_response(self, body: dict) -> EmbeddingResponse:
        usage = usage_in(body.get("meta"))
        return EmbeddingResponse(
            model=self.model,
            vectors=body["embeddings"][FLOAT],
            input_tokens=usage.input_tokens if usage else None,
        )
