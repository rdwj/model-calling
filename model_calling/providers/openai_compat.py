"""
Backends that speak OpenAI Chat Completions at `{base_url}/chat/completions`:
OpenAI itself, vLLM, and OpenRouter.

`base_url` is the API root including the version, as in the official
`openai` SDK (`https://api.openai.com/v1`, `http://localhost:8000/v1`). The
subclasses differ only in fields OpenAI doesn't define: where reasoning goes
on assistant turns, `top_k`, and which max-tokens field to send. Reasoning
is read from `reasoning` (vLLM 0.24, OpenRouter) or `reasoning_content`
(older vLLM, other servers).
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
from model_calling.core.embeddings import EmbeddingRequest, EmbeddingResponse
from model_calling.formats import TranslationError
from model_calling.formats.embeddings import FORMAT as EMBEDDINGS_FORMAT
from model_calling.formats.chat_completions import (
    FORMAT,
    REASONING_FIELDS,
    STOP_REASONS,
    tool_call_in,
    usage_in,
)
from model_calling.errors import APIError
from model_calling.providers.base import Provider, StreamParser, new_id
from model_calling.providers.openai_compat_stream import ChatCompletionsStreamParser

logger = logging.getLogger(__name__)

CHAT_PATH = "/chat/completions"
EMBEDDINGS_PATH = "/embeddings"
# Ingress fields this provider sets itself rather than passing through.
_MANAGED_FIELDS = ("stream", "stream_options")
_TOOL_CHOICE_MODES = {"auto": "auto", "none": "none", "any": "required"}


def _text(blocks: list, sep: str = "") -> str:
    return sep.join(b.text for b in blocks if isinstance(b, Text))


def _check_part(block: Any, provider: str) -> None:
    if isinstance(block, Opaque) and block.format != FORMAT:
        raise TranslationError(
            f"{provider} cannot accept a {block.format} {block.data.get('type')!r} block"
        )
    if isinstance(block, Image) and isinstance(block.source, FileSource):
        raise TranslationError(f"{provider} cannot accept an image by file id")


def _part_out(block: Any) -> dict:
    if isinstance(block, Text):
        return {"type": "text", "text": block.text}
    if isinstance(block, Image):
        source = block.source
        if isinstance(source, Base64Source):
            url = f"data:{source.media_type};base64,{source.data}"
        else:
            url = source.url
        return {"type": "image_url", "image_url": {"url": url, **block.extras_for(FORMAT)}}
    return block.data


def _content_out(blocks: list, shorthand: bool) -> str | list:
    """A string when the content is plain text, otherwise content parts."""
    if all(isinstance(b, Text) for b in blocks) and (shorthand or len(blocks) <= 1):
        return _text(blocks)
    return [_part_out(b) for b in blocks]


def _tool_call_out(block: ToolCall | Opaque) -> dict:
    if isinstance(block, Opaque):
        return block.data
    arguments = block.raw_arguments if block.raw_arguments is not None else json.dumps(block.input)
    return {
        "id": block.id,
        "type": "function",
        "function": {"name": block.name, "arguments": arguments},
    }


class OpenAICompatProvider(Provider):
    name = "openai_compat"
    # Field for thinking on assistant turns, or None to drop it.
    reasoning_field: ClassVar[str | None] = "reasoning"
    supports_top_k: ClassVar[bool] = True
    max_tokens_field: ClassVar[str] = "max_tokens"

    def endpoint(self) -> str:
        return f"{self.base_url}{CHAT_PATH}"

    # --- request ----------------------------------------------------------

    def unsupported_params(self, request: Request) -> list[str]:
        names = []
        thinking = request.thinking
        if thinking is not None and thinking.budget_tokens is not None and thinking.effort is None:
            # Chat Completions takes reasoning_effort; a token budget has no equivalent.
            names.append("thinking.budget_tokens")
        return names

    def build_request(self, request: Request) -> dict:
        messages: list[dict] = []
        if request.system:
            # One block per client system message; keep them apart.
            messages.append({"role": "system", "content": _text(request.system, "\n")})
        for msg in request.messages:
            messages.extend(self._message_out(msg))
        body: dict[str, Any] = {
            k: v for k, v in request.extras_for(FORMAT).items() if k not in _MANAGED_FIELDS
        }
        body.update(model=self.model, messages=messages)
        if request.tools:
            body["tools"] = [self._tool_out(t) for t in request.tools]
        if request.tool_choice is not None:
            body.update(self._tool_choice_out(request))
        params = {
            self.max_tokens_field: request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "seed": request.seed,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "stop": request.stop,
            "reasoning_effort": self._reasoning_effort(request),
            "response_format": self._response_format_out(request),
        }
        body.update({k: v for k, v in params.items() if v is not None})
        # Checked after the merge: a Chat Completions client's top_k arrives in extras.
        if "top_k" in body and not self.supports_top_k:
            raise TranslationError(f"{self.name} does not support top_k")
        if request.stream:
            # Usage arrives only when asked for; the ingress decides whether
            # the client sees it.
            body["stream_options"] = {"include_usage": True}
        return body

    def _message_out(self, msg: Message) -> list[dict]:
        """One internal turn as Chat Completions messages: a `tool` message per
        tool result, then the rest of the turn."""
        for block in msg.content:
            _check_part(block, self.name)
        results = [self._tool_result_out(b) for b in msg.content if isinstance(b, ToolResult)]
        if msg.role == "assistant":
            return results + [self._assistant_out(msg)]
        rest = [b for b in msg.content if not isinstance(b, ToolResult)]
        if results and not rest:
            return results
        role = "system" if msg.role == "system" else "user"
        if role == "system" and not all(isinstance(b, Text) for b in rest):
            raise TranslationError("a system message may contain only text")
        return results + [{"role": role, "content": _content_out(rest, msg.content_shorthand)}]

    def _assistant_out(self, msg: Message) -> dict:
        calls, parts = [], []
        for b in msg.content:
            # Chat Completions tool calls of a type we don't model carry an id;
            # unknown content parts don't.
            if isinstance(b, ToolCall) or (isinstance(b, Opaque) and "id" in b.data):
                calls.append(b)
            elif isinstance(b, (Text, Image, Opaque)):
                parts.append(b)
        out: dict[str, Any] = {"role": "assistant"}
        if parts:
            out["content"] = _content_out(parts, msg.content_shorthand)
        elif not calls:
            out["content"] = ""
        else:
            out["content"] = None
        reasoning = "".join(b.thinking for b in msg.content if isinstance(b, Thinking))
        if reasoning and self.reasoning_field:
            out[self.reasoning_field] = reasoning
        elif reasoning or any(isinstance(b, RedactedThinking) for b in msg.content):
            logger.debug("%s takes no reasoning on assistant turns; dropping it", self.name)
        if calls:
            out["tool_calls"] = [_tool_call_out(c) for c in calls]
        return out

    def _tool_result_out(self, result: ToolResult) -> dict:
        content = result.content or []
        if not all(isinstance(b, Text) for b in content):
            raise TranslationError(f"{self.name} tool results may contain only text")
        return {"role": "tool", "tool_call_id": result.tool_call_id, "content": _text(content)}

    def _tool_out(self, tool: Tool | Opaque) -> dict:
        if isinstance(tool, Opaque):
            if tool.format != FORMAT:
                raise TranslationError(
                    f"{self.name} cannot run {tool.format} tool {tool.data.get('type')!r}"
                )
            return tool.data
        fn: dict[str, Any] = {"name": tool.name}
        if tool.description is not None:
            fn["description"] = tool.description
        fn["parameters"] = tool.input_schema
        fn.update(tool.extras_for(FORMAT))
        return {"type": "function", "function": fn}

    @staticmethod
    def _tool_choice_out(request: Request) -> dict:
        choice = request.tool_choice
        out: dict[str, Any] = {}
        if choice.mode == "tool":
            out["tool_choice"] = {"type": "function", "function": {"name": choice.name}}
        elif choice.mode != "auto" or request.tools:
            out["tool_choice"] = _TOOL_CHOICE_MODES[choice.mode]
        if choice.parallel is not None:
            out["parallel_tool_calls"] = choice.parallel
        return out

    def _reasoning_effort(self, request: Request) -> str | None:
        thinking = request.thinking
        if thinking is None:
            return None
        if thinking.mode == "disabled":
            return "none"
        if thinking.effort is None:
            logger.debug("%s takes reasoning effort, not a token budget; not sent", self.name)
        return thinking.effort

    @staticmethod
    def _response_format_out(request: Request) -> dict | None:
        fmt = request.response_format
        if fmt is None:
            return None
        if fmt.type != "json_schema":
            return {"type": fmt.type}
        schema: dict[str, Any] = {"name": fmt.name or "response"}
        if fmt.json_schema is not None:
            schema["schema"] = fmt.json_schema
        if fmt.strict is not None:
            schema["strict"] = fmt.strict
        return {"type": "json_schema", "json_schema": schema}

    # --- response ---------------------------------------------------------

    def parse_response(self, body: dict) -> Response:
        if body.get("error") and not body.get("choices"):
            # OpenRouter reports some upstream failures inside an HTTP 200.
            raise APIError(
                f"{self.name} returned an error: {self.error_message(body)}",
                provider=self.name,
                body=body,
            )
        choices = body.get("choices")
        if not choices:
            raise TranslationError("response has no choices")
        choice = choices[0]
        msg = choice.get("message") or {}
        blocks: list = []
        reasoning = next((msg[f] for f in REASONING_FIELDS if msg.get(f)), None)
        if reasoning:
            blocks.append(Thinking(thinking=reasoning))
        if msg.get("content"):
            blocks.append(Text(text=msg["content"]))
        if msg.get("refusal"):
            blocks.append(Text(text=msg["refusal"]))
        calls = [tool_call_in(tc) for tc in msg.get("tool_calls") or []]
        blocks.extend(calls)
        stop_reason = STOP_REASONS.get(choice.get("finish_reason"))
        if calls and stop_reason == "end_turn":
            stop_reason = "tool_use"
        if msg.get("refusal"):
            stop_reason = "refusal"
        return Response(
            id=body.get("id") or new_id(),
            model=body.get("model") or self.model,
            content=blocks,
            stop_reason=stop_reason,
            usage=usage_in(body.get("usage")),
        )

    def stream_parser(self) -> StreamParser:
        return ChatCompletionsStreamParser(self.name, self.model)

    # --- embeddings -------------------------------------------------------

    def embedding_endpoint(self) -> str:
        return f"{self.base_url}{EMBEDDINGS_PATH}"

    def build_embedding_request(self, request: EmbeddingRequest) -> dict:
        # Floats always; formats.embeddings applies a client's base64.
        body: dict[str, Any] = {
            **request.extras_for(EMBEDDINGS_FORMAT),
            "model": self.model,
            "input": request.input,
        }
        if request.dimensions is not None:
            body["dimensions"] = request.dimensions
        if request.user is not None:
            body["user"] = request.user
        return body

    def parse_embedding_response(self, body: dict) -> EmbeddingResponse:
        data = sorted(body["data"], key=lambda item: item["index"])
        return EmbeddingResponse(
            model=body.get("model", self.model),
            vectors=[item["embedding"] for item in data],
            input_tokens=(body.get("usage") or {}).get("prompt_tokens"),
        )


class OpenAIProvider(OpenAICompatProvider):
    name = "openai"
    reasoning_field = None
    supports_top_k = False
    # OpenAI's reasoning models reject max_tokens.
    max_tokens_field = "max_completion_tokens"


class VLLMProvider(OpenAICompatProvider):
    name = "vllm"


class OpenRouterProvider(OpenAICompatProvider):
    name = "openrouter"
