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


def _signed(messages: list[Message]) -> list[Message]:
    """`messages` without unsigned thinking on assistant turns (from Ollama,
    or a Chat Completions backend), which Anthropic rejects when sent back.
    A turn left empty is dropped: Anthropic rejects empty content, and
    merges the consecutive user turns that dropping it can leave. Thinking
    another server signed (vLLM signs its own) can't be told apart and is
    kept."""
    out = []
    for msg in messages:
        if msg.role == "assistant":
            kept = [b for b in msg.content if not (isinstance(b, Thinking) and not b.signature)]
            if len(kept) != len(msg.content):
                logger.debug("dropping unsigned thinking from an assistant turn")
                if not kept:
                    continue
                msg = msg.model_copy(update={"content": kept})
        out.append(msg)
    return out


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
        # Chat Completions extras that Anthropic doesn't support.
        chat_extras = request.extras_for("openai_chat")
        for name in ("logprobs", "logit_bias", "user"):
            if chat_extras.get(name) is not None:
                names.append(name)
        fmt = request.response_format
        if fmt is not None and fmt.type == "json_object":
            # Messages constrains output only to a schema.
            names.append("response_format")
        thinking = request.thinking
        if thinking is not None and thinking.effort is not None and thinking.effort not in EFFORTS:
            names.append("thinking.effort")
        return names

    def build_request(self, request: Request) -> dict:
        update: dict[str, Any] = {
            "model": self.model,
            "messages": _signed(request.messages),
            "thinking": self._thinking_out(request),
        }
        choice = request.tool_choice
        if choice is not None and choice.mode == "auto" and not request.tools:
            # A Chat Completions parallel_tool_calls with no tools to call.
            update["tool_choice"] = None
        if request.max_tokens is None:
            update["max_tokens"] = DEFAULT_MAX_TOKENS
        return anthropic.request_from_internal(request.model_copy(update=update))

    @staticmethod
    def _thinking_out(request: Request):
        """The thinking config to send. An effort with no token budget (a
        Chat Completions `reasoning_effort`) is adaptive thinking at that
        effort. An effort Messages doesn't know (OpenAI's "minimal") is
        dropped, and with no budget to fall back on, so is the thinking:
        the client asked for little of it, not the model's default."""
        thinking = request.thinking
        if thinking is None or thinking.effort is None:
            return thinking
        if thinking.effort not in EFFORTS:
            if thinking.mode == "enabled" and thinking.budget_tokens is None:
                return None
            return thinking.model_copy(update={"effort": None})
        if thinking.mode == "enabled" and thinking.budget_tokens is None:
            return thinking.model_copy(update={"mode": "adaptive"})
        return thinking

    # --- response ---------------------------------------------------------

    def parse_response(self, body: dict) -> Response:
        response = anthropic.response_to_internal(body)
        if response.stop_reason == "end_turn" and any(
            isinstance(b, ToolCall) for b in response.content
        ):
            # vLLM ends a forced tool call with end_turn; Anthropic says tool_use.
            response.stop_reason = "tool_use"
        # Pass through stop_details and container from the wire.
        if body.get("stop_details") is not None:
            response.stop_details = body["stop_details"]
        if body.get("container") is not None:
            response.container = body["container"]
        return response

    def stream_parser(self) -> StreamParser:
        return AnthropicStreamParser(self.name, self.model)
