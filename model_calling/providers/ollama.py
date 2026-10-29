"""
Ollama's native `/api/chat`, including its NDJSON stream and `thinking` field.

Ollama sends each tool call whole, without streaming its arguments, and may
omit the call id. The stream parser invents an id when needed and emits the
arguments as a single `ToolInputDelta`.
"""

import json
import logging
from typing import Any

from model_calling.core import (
    Base64Source,
    BlockDelta,
    BlockStart,
    BlockStop,
    Image,
    Message,
    MessageDelta,
    MessageStart,
    MessageStop,
    Opaque,
    Request,
    Response,
    StreamError,
    StreamEvent,
    Text,
    TextDelta,
    Thinking,
    ThinkingDelta,
    Tool,
    ToolCall,
    ToolInputDelta,
    ToolResult,
    Usage,
)
from model_calling.core.embeddings import EmbeddingRequest, EmbeddingResponse
from model_calling.formats import TranslationError
from model_calling.formats.embeddings import FORMAT as EMBEDDINGS_FORMAT
from model_calling.providers.base import (
    BAD_GATEWAY,
    Provider,
    StreamParser,
    new_id,
)

logger = logging.getLogger(__name__)

CHAT_PATH = "/api/chat"
EMBED_PATH = "/api/embed"
# Model metadata, including which `think` values the model takes.
SHOW_PATH = "/api/show"
# Ollama has no tool_choice, so it can't make the model call a tool.
FORCED_TOOL_CHOICES = ("any", "tool")
# Ollama's done_reason -> internal stop reason. Ollama reports "stop" even
# when the turn ends in tool calls, so callers check for those first.
_STOP_REASONS = {"stop": "end_turn", "length": "max_tokens"}


def _text(blocks: list) -> str:
    return "\n".join(b.text for b in blocks if isinstance(b, Text))


def _usage(chunk: dict) -> Usage | None:
    if "prompt_eval_count" not in chunk and "eval_count" not in chunk:
        return None
    return Usage(
        input_tokens=chunk.get("prompt_eval_count"),
        output_tokens=chunk.get("eval_count"),
        cache_read_tokens=chunk.get("prompt_eval_cached_count"),
    )


def _stop_reason(chunk: dict, saw_tool_call: bool) -> str | None:
    if saw_tool_call:
        return "tool_use"
    return _STOP_REASONS.get(chunk.get("done_reason", "stop"))


def _tool_call_in(raw: dict) -> ToolCall:
    fn = raw["function"]
    arguments = fn.get("arguments") or {}
    call_id = raw.get("id") or new_id("call_")
    parsed = arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            parsed = None
    if isinstance(parsed, dict):
        return ToolCall(id=call_id, name=fn["name"], input=parsed)
    raw_arguments = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ToolCall(id=call_id, name=fn["name"], input={}, raw_arguments=raw_arguments)


def _strict(tool: Any) -> bool:
    """Whether a tool asks for strict schema adherence, which Ollama can't give."""
    return isinstance(tool, Tool) and bool(tool.strict)


class OllamaProvider(Provider):
    """`think` takes a bool, or a named level for models that define levels
    (gpt-oss: "low", "medium", "high"). Which ones a model takes comes from
    `/api/show`, looked up only for a request with a thinking effort."""

    name = "ollama"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # The model's named thinking levels, once looked up.
        self.think_levels: tuple[str, ...] = ()

    def endpoint(self) -> str:
        return f"{self.base_url}{CHAT_PATH}"

    def needs_model_info(self, request: Request) -> bool:
        thinking = request.thinking
        return thinking is not None and thinking.mode != "disabled" and thinking.effort is not None

    def model_info_request(self) -> tuple[str, dict]:
        return f"{self.base_url}{SHOW_PATH}", {"model": self.model}

    def read_model_info(self, body: Any) -> tuple[str, ...]:
        """The named levels in /api/show's `thinking.values`; booleans there
        are on/off controls, not levels."""
        thinking = body.get("thinking") if isinstance(body, dict) else None
        values = thinking.get("values") if isinstance(thinking, dict) else None
        if not isinstance(values, list):
            return ()
        return tuple(v for v in values if isinstance(v, str))

    def use_model_info(self, info: tuple[str, ...] | None) -> None:
        self.think_levels = info or ()

    # --- request ----------------------------------------------------------

    def unsupported_params(self, request: Request) -> list[str]:
        names = []
        choice = request.tool_choice
        if choice is not None and choice.mode in FORCED_TOOL_CHOICES:
            names.append("tool_choice")
        sends_tools = bool(request.tools) and not (choice and choice.mode == "none")
        if sends_tools and choice is not None and choice.parallel is False:
            # Ollama has no switch for it; the model may still call several tools.
            names.append("parallel_tool_calls")
        if any(_strict(t) for t in request.tools or ()):
            names.append("tools.strict")
        return names

    def build_request(self, request: Request) -> dict:
        tool_names: dict[str, str] = {}
        messages: list[dict] = []
        if request.system:
            messages.append({"role": "system", "content": _text(request.system)})
        for msg in request.messages:
            messages.extend(self._message_out(msg, tool_names))
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        choice = request.tool_choice
        if choice is not None and choice.mode in FORCED_TOOL_CHOICES:
            # Ollama has no tool_choice; the model may still answer in text.
            logger.warning("Ollama cannot force a tool call; ignoring tool_choice %r", choice.mode)
        if request.tools and not (choice and choice.mode == "none"):
            body["tools"] = [self._tool_out(t) for t in request.tools]
        options = {
            "num_predict": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "seed": request.seed,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "stop": request.stop,
        }
        options = {k: v for k, v in options.items() if v is not None}
        if options:
            body["options"] = options
        if request.thinking is not None:
            body["think"] = self._think(request.thinking)
        fmt = request.response_format
        if fmt is not None and fmt.type == "json_object":
            body["format"] = "json"
        elif fmt is not None and fmt.type == "json_schema":
            body["format"] = fmt.json_schema or "json"
        return body

    def _think(self, thinking: Any) -> bool | str:
        """The effort as a level when the model has it, otherwise on or off.
        A model without levels thinks at its own pace."""
        if thinking.mode != "disabled" and thinking.effort in self.think_levels:
            return thinking.effort
        if thinking.mode != "disabled" and thinking.effort is not None:
            logger.debug(
                "%s has no thinking level %r; sending think=true", self.model, thinking.effort
            )
        return thinking.mode != "disabled"

    def _tool_out(self, tool: Any) -> dict:
        if not isinstance(tool, Tool):
            raise TranslationError(
                f"Ollama cannot run {tool.format} tool {tool.data.get('type')!r}"
            )
        fn = {"name": tool.name, "parameters": tool.input_schema}
        if tool.description is not None:
            fn["description"] = tool.description
        return {"type": "function", "function": fn}

    def _message_out(self, msg: Message, tool_names: dict[str, str]) -> list[dict]:
        """One internal turn as Ollama messages: a `tool` message per tool result,
        then the turn itself."""
        for block in msg.content:
            if isinstance(block, Opaque):
                raise TranslationError(
                    f"Ollama cannot accept a {block.format} {block.data.get('type')!r} block"
                )
            if isinstance(block, Image) and not isinstance(block.source, Base64Source):
                raise TranslationError("Ollama accepts only base64 images, not image URLs")
        calls = [b for b in msg.content if isinstance(b, ToolCall)]
        tool_names.update({c.id: c.name for c in calls})
        out = [
            self._tool_result_out(b, tool_names) for b in msg.content if isinstance(b, ToolResult)
        ]
        text = _text(msg.content)
        images = [b.source.data for b in msg.content if isinstance(b, Image)]
        thinking = "".join(b.thinking for b in msg.content if isinstance(b, Thinking))
        if out and not (text or images or thinking or calls):
            return out
        turn: dict[str, Any] = {"role": msg.role, "content": text}
        if images:
            turn["images"] = images
        if thinking:
            turn["thinking"] = thinking
        if calls:
            turn["tool_calls"] = [
                {"function": {"name": c.name, "arguments": c.input}} for c in calls
            ]
        return out + [turn]

    @staticmethod
    def _tool_result_out(result: ToolResult, tool_names: dict[str, str]) -> dict:
        tool_msg = {"role": "tool", "content": _text(result.content or [])}
        if result.tool_call_id in tool_names:
            tool_msg["tool_name"] = tool_names[result.tool_call_id]
        return tool_msg

    # --- response ---------------------------------------------------------

    def parse_response(self, body: dict) -> Response:
        msg = body.get("message") or {}
        blocks: list = []
        if msg.get("thinking"):
            blocks.append(Thinking(thinking=msg["thinking"]))
        if msg.get("content"):
            blocks.append(Text(text=msg["content"]))
        calls = [_tool_call_in(tc) for tc in msg.get("tool_calls") or []]
        blocks.extend(calls)
        return Response(
            id=new_id(),
            model=body.get("model", self.model),
            content=blocks,
            stop_reason=_stop_reason(body, bool(calls)),
            usage=_usage(body),
        )

    def stream_parser(self) -> StreamParser:
        return _OllamaStreamParser(self.model)

    # --- embeddings -------------------------------------------------------

    def embedding_endpoint(self) -> str:
        return f"{self.base_url}{EMBED_PATH}"

    def unsupported_embedding_params(self, request: EmbeddingRequest) -> list[str]:
        # `user` only labels the caller, so it is dropped silently, as for chat.
        return sorted(request.extras_for(EMBEDDINGS_FORMAT))

    def build_embedding_request(self, request: EmbeddingRequest) -> dict:
        if not all(isinstance(x, str) for x in request.input):
            raise TranslationError("Ollama embeds text only, not token arrays")
        body: dict[str, Any] = {"model": self.model, "input": request.input}
        if request.dimensions is not None:
            body["dimensions"] = request.dimensions
        return body

    def parse_embedding_response(self, body: dict) -> EmbeddingResponse:
        return EmbeddingResponse(
            model=body.get("model", self.model),
            vectors=body["embeddings"],
            input_tokens=body.get("prompt_eval_count"),
        )


class _OllamaStreamParser(StreamParser):
    def __init__(self, model: str):
        self.model = model
        self.started = False
        self.index = -1
        # Kind of the open block ("thinking" or "text"), if any.
        self.open: str | None = None
        self.saw_tool_call = False
        self.done = False

    def feed(self, line: str) -> list[StreamEvent]:
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            return [
                StreamError(status=BAD_GATEWAY, message=f"Ollama sent a non-JSON line: {line!r}")
            ]
        if "error" in chunk:
            return [StreamError(status=BAD_GATEWAY, message=f"Ollama error: {chunk['error']}")]
        events: list[StreamEvent] = []
        if not self.started:
            self.started = True
            events.append(MessageStart(id=new_id(), model=chunk.get("model", self.model)))
        msg = chunk.get("message") or {}
        if msg.get("thinking"):
            events += self._into("thinking", ThinkingDelta(thinking=msg["thinking"]))
        if msg.get("content"):
            events += self._into("text", TextDelta(text=msg["content"]))
        for raw in msg.get("tool_calls") or []:
            events += self._tool_call(_tool_call_in(raw))
        if chunk.get("done"):
            self.done = True
            events += self._close()
            events.append(
                MessageDelta(
                    stop_reason=_stop_reason(chunk, self.saw_tool_call), usage=_usage(chunk)
                )
            )
            events.append(MessageStop())
        return events

    def close(self) -> list[StreamEvent]:
        if self.done:
            return []
        return [
            StreamError(status=BAD_GATEWAY, message="Ollama closed the stream before it finished")
        ]

    def _close(self) -> list[StreamEvent]:
        if self.open is None:
            return []
        self.open = None
        return [BlockStop(index=self.index)]

    def _into(self, kind: str, delta: Any) -> list[StreamEvent]:
        """Route a delta into an open block of `kind`, opening one if needed."""
        events: list[StreamEvent] = []
        if self.open != kind:
            events += self._close()
            self.index += 1
            self.open = kind
            block = Thinking(thinking="") if kind == "thinking" else Text(text="")
            events.append(BlockStart(index=self.index, block=block))
        events.append(BlockDelta(index=self.index, delta=delta))
        return events

    def _tool_call(self, call: ToolCall) -> list[StreamEvent]:
        self.saw_tool_call = True
        events = self._close()
        self.index += 1
        arguments = call.raw_arguments if call.raw_arguments is not None else json.dumps(call.input)
        events += [
            BlockStart(
                index=self.index, block=call.model_copy(update={"input": {}, "raw_arguments": None})
            ),
            BlockDelta(index=self.index, delta=ToolInputDelta(partial_json=arguments)),
            BlockStop(index=self.index),
        ]
        return events
