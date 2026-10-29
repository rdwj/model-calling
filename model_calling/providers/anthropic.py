"""
Backends that speak Anthropic Messages at `{base_url}/messages`: Anthropic
itself, OpenRouter's `/api/v1/messages`, and vLLM's native `/v1/messages`.

`base_url` is the API root including the version
(`https://api.anthropic.com/v1`), as for the OpenAI-compatible providers.
The body is `formats.anthropic.request_from_internal`, so a Messages client's
request reaches the backend unchanged. Auth is a bearer token, which is how
the `anthropic` 2.0.0 SDK sends an API key and what OpenRouter and vLLM
take; `anthropic-version` is the version that SDK sends.
"""

import logging
from typing import Any

from model_calling.core import Message, Request, Response, Thinking, ToolCall
from model_calling.formats import anthropic
from model_calling.providers.anthropic_stream import AnthropicStreamParser
from model_calling.providers.base import Provider, StreamParser

logger = logging.getLogger(__name__)

MESSAGES_PATH = "/messages"
# `anthropic.ANTHROPIC_API_VERSION` in the pinned SDK.
API_VERSION = "2026-10-21"
# Messages requires max_tokens; Chat Completions clients may leave it out.
DEFAULT_MAX_TOKENS = 4096
# `output_config.effort` values Messages accepts.
EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


def _signed(msg: Message) -> Message:
    """`msg` without thinking no Anthropic model signed (from Ollama, or a
    Chat Completions backend), which Anthropic rejects when sent back."""
    kept = [b for b in msg.content if not (isinstance(b, Thinking) and not b.signature)]
    if len(kept) == len(msg.content):
        return msg
    logger.debug("dropping unsigned thinking from an assistant turn")
    return msg.model_copy(update={"content": kept})


class AnthropicProvider(Provider):
    name = "anthropic"

    def endpoint(self) -> str:
        return f"{self.base_url}{MESSAGES_PATH}"

    def headers(self) -> dict[str, str]:
        return {"anthropic-version": API_VERSION, **super().headers()}

    # --- request ----------------------------------------------------------

    def unsupported_params(self, request: Request) -> list[str]:
        names = [
            name
            for name in ("seed", "presence_penalty", "frequency_penalty")
            if getattr(request, name) is not None
        ]
        fmt = request.response_format
        if fmt is not None and fmt.type == "json_object":
            # Messages constrains output only to a schema.
            names.append("response_format")
        thinking = request.thinking
        if thinking is not None and thinking.effort is not None and thinking.effort not in EFFORTS:
            names.append("reasoning_effort")
        return names

    def build_request(self, request: Request) -> dict:
        update: dict[str, Any] = {
            "model": self.model,
            "messages": [_signed(m) if m.role == "assistant" else m for m in request.messages],
            "thinking": self._thinking_out(request),
        }
        if request.max_tokens is None:
            update["max_tokens"] = DEFAULT_MAX_TOKENS
        return anthropic.request_from_internal(request.model_copy(update=update))

    @staticmethod
    def _thinking_out(request: Request):
        """The thinking config to send. An effort with no token budget (a
        Chat Completions `reasoning_effort`) is adaptive thinking at that
        effort; an effort Messages doesn't know is dropped."""
        thinking = request.thinking
        if thinking is None or thinking.effort is None:
            return thinking
        update: dict[str, Any] = {}
        if thinking.mode == "enabled" and thinking.budget_tokens is None:
            update["mode"] = "adaptive"
        if thinking.effort not in EFFORTS:
            update["effort"] = None
        return thinking.model_copy(update=update) if update else thinking

    # --- response ---------------------------------------------------------

    def parse_response(self, body: dict) -> Response:
        response = anthropic.response_to_internal(body)
        if response.stop_reason == "end_turn" and any(
            isinstance(b, ToolCall) for b in response.content
        ):
            # vLLM ends a forced tool call with end_turn; Anthropic says tool_use.
            response.stop_reason = "tool_use"
        return response

    def stream_parser(self) -> StreamParser:
        return AnthropicStreamParser(self.name, self.model)
