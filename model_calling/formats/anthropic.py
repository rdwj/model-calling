"""
Anthropic Messages wire format to and from the internal format.

Fields the internal format doesn't model go into `extras["anthropic"]` and
block types it doesn't model become `Opaque` blocks, so an Anthropic →
internal → Anthropic round trip is lossless. Modeled fields that arrive as an
explicit null, or with a value newer than this module (an unknown stop
reason or cache TTL), stay in extras verbatim. On the way out extras are
written first and non-null typed fields override them, so a typed field set
after decoding always wins.
"""

import copy
from typing import Any, Callable, TypeVar

from pydantic import ValidationError

from model_calling.core import (
    Base64Source,
    CacheControl,
    FileSource,
    Image,
    Message,
    Opaque,
    RedactedThinking,
    Request,
    Response,
    Text,
    Thinking,
    ThinkingConfig,
    Tool,
    ToolCall,
    ToolChoice,
    ToolResult,
    Usage,
    UrlSource,
)
from model_calling.core.base import CoreModel, Extras
from model_calling.formats import TranslationError

FORMAT = "anthropic"

# Wire stop reason -> internal, where the names differ.
_STOP_REASON_IN = {"model_context_window_exceeded": "context_window_exceeded"}
_STOP_REASON_OUT = {v: k for k, v in _STOP_REASON_IN.items()}
_KNOWN_STOP_REASONS = {
    "end_turn",
    "max_tokens",
    "stop_sequence",
    "tool_use",
    "pause_turn",
    "refusal",
    "model_context_window_exceeded",
}
_THINKING_MODES = {"enabled", "adaptive", "disabled"}
_TOOL_CHOICE_MODES = {"auto", "any", "tool", "none"}
_CACHE_TTLS = {None, "5m", "1h"}
_TOOL_RESULT_CONTENT_TYPES = {"text", "image"}
_IMAGE_SOURCES = {"base64": Base64Source, "url": UrlSource, "file": FileSource}
# The signature sent for thinking that no Anthropic model signed (Ollama,
# vLLM's Chat Completions), as Anthropic itself does in content_block_start.
# The SDK requires the field. Ingress reads it back as unsigned.
UNSIGNED = ""

T = TypeVar("T")


def _extras(rest: dict) -> Extras:
    return {FORMAT: rest} if rest else {}


def _take(rest: dict, key: str) -> Any:
    """Pop a modeled field. An explicit null stays in `rest` and so in extras."""
    return None if rest.get(key) is None else rest.pop(key)


def _out(model: CoreModel, fields: dict) -> dict:
    """This format's extras, overridden by every non-null typed field."""
    out = dict(model.extras_for(FORMAT))
    out.update({k: v for k, v in fields.items() if v is not None})
    return out


def _translating(what: str, fn: Callable[[], T]) -> T:
    """Run a translation, turning malformed input into one TranslationError."""
    try:
        return fn()
    except TranslationError:
        raise
    except (KeyError, TypeError, AttributeError, ValidationError) as e:
        raise TranslationError(f"invalid Anthropic {what}: {type(e).__name__}: {e}") from e


# --- blocks -------------------------------------------------------------


def _cache_in(rest: dict) -> CacheControl | None:
    raw = rest.get("cache_control")
    if (
        not isinstance(raw, dict)
        or raw.get("type") != "ephemeral"
        or raw.get("ttl") not in _CACHE_TTLS
    ):
        return None
    cc = dict(rest.pop("cache_control"))
    del cc["type"]
    return CacheControl(ttl=_take(cc, "ttl"), extras=_extras(cc))


def _cache_out(cc: CacheControl | None) -> dict | None:
    return None if cc is None else _out(cc, {"type": cc.type, "ttl": cc.ttl})


def _block_in(raw: dict, allowed: set[str] | None = None) -> Any:
    kind = raw.get("type")
    if (allowed is not None and kind not in allowed) or kind not in _BLOCK_IN:
        return Opaque(format=FORMAT, data=raw)
    if kind == "image" and raw.get("source", {}).get("type") not in _IMAGE_SOURCES:
        return Opaque(format=FORMAT, data=raw)
    rest = dict(raw)
    del rest["type"]
    cache_control = _cache_in(rest)
    block = _BLOCK_IN[kind](rest)
    block.cache_control = cache_control
    block.extras = _extras(rest)
    return block


def _image_in(rest: dict) -> Image:
    src = dict(rest.pop("source"))
    cls = _IMAGE_SOURCES[src.pop("type")]
    fields = {name: src.pop(name) for name in cls.model_fields if name not in ("type", "extras")}
    return Image(source=cls(**fields, extras=_extras(src)))


def _tool_result_in(rest: dict) -> ToolResult:
    content = _take(rest, "content")
    shorthand = isinstance(content, str)
    if shorthand:
        blocks = [Text(text=content)]
    elif content is None:
        blocks = None
    else:
        blocks = [_block_in(b, _TOOL_RESULT_CONTENT_TYPES) for b in content]
    return ToolResult(
        tool_call_id=rest.pop("tool_use_id"),
        content=blocks,
        content_shorthand=shorthand,
        is_error=_take(rest, "is_error"),
    )


_BLOCK_IN = {
    "text": lambda r: Text(text=r.pop("text")),
    "image": _image_in,
    "tool_use": lambda r: ToolCall(id=r.pop("id"), name=r.pop("name"), input=r.pop("input")),
    "tool_result": _tool_result_in,
    "thinking": lambda r: Thinking(
        thinking=r.pop("thinking"), signature=_take(r, "signature") or None
    ),
    "redacted_thinking": lambda r: RedactedThinking(data=r.pop("data")),
}


def _block_out(block: Any) -> dict:
    if isinstance(block, Opaque):
        if block.format != FORMAT:
            raise TranslationError(
                f"cannot send a {block.format!r} block of type "
                f"{block.data.get('type')!r} in Anthropic format"
            )
        return copy.deepcopy(block.data)
    fields: dict[str, Any] = {"cache_control": _cache_out(block.cache_control)}
    if isinstance(block, Text):
        fields |= {"type": "text", "text": block.text}
    elif isinstance(block, Image):
        src = block.source
        fields |= {"type": "image", "source": _out(src, src.model_dump(exclude={"extras"}))}
    elif isinstance(block, ToolCall):
        fields |= {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    elif isinstance(block, ToolResult):
        fields |= {
            "type": "tool_result",
            "tool_use_id": block.tool_call_id,
            "content": (
                None
                if block.content is None
                else _content_out(block.content, block.content_shorthand)
            ),
            "is_error": block.is_error,
        }
    elif isinstance(block, Thinking):
        fields |= {
            "type": "thinking",
            "thinking": block.thinking,
            "signature": block.signature or UNSIGNED,
        }
    elif isinstance(block, RedactedThinking):
        fields |= {"type": "redacted_thinking", "data": block.data}
    else:
        raise TranslationError(f"no Anthropic encoding for {type(block).__name__}")
    return _out(block, fields)


def block_from_internal(block: Any) -> dict:
    """One internal content block as an Anthropic content block."""
    return _translating("content block", lambda: _block_out(block))


def _is_plain_text(blocks: list) -> bool:
    """True if `blocks` can be written as the bare-string content shorthand."""
    return (
        len(blocks) == 1
        and isinstance(blocks[0], Text)
        and blocks[0].cache_control is None
        and not blocks[0].extras
    )


def _content_in(content: str | list) -> tuple[list, bool]:
    if isinstance(content, str):
        return [Text(text=content)], True
    return [_block_in(b) for b in content], False


def _content_out(blocks: list, shorthand: bool) -> str | list:
    if shorthand and _is_plain_text(blocks):
        return blocks[0].text
    return [_block_out(b) for b in blocks]


# --- request ------------------------------------------------------------


def _system_text(content: list, where: str) -> list:
    if not all(isinstance(b, Text) for b in content):
        raise TranslationError(f"Anthropic {where} may contain only text blocks")
    return content


def _message_in(raw: dict) -> Message:
    rest = dict(raw)
    role = rest.pop("role")
    content, shorthand = _content_in(rest.pop("content"))
    if role == "system":
        _system_text(content, "'system' messages")
    return Message(role=role, content=content, content_shorthand=shorthand, extras=_extras(rest))


def _message_out(msg: Message) -> dict:
    return _out(
        msg, {"role": msg.role, "content": _content_out(msg.content, msg.content_shorthand)}
    )


def _tool_in(raw: dict) -> Tool | Opaque:
    # Custom tools have an input_schema and no type, or type "custom"; anything
    # else is a server tool (web search, code execution, ...) kept verbatim.
    if "input_schema" not in raw or raw.get("type", "custom") != "custom":
        return Opaque(format=FORMAT, data=raw)
    rest = dict(raw)
    cache_control = _cache_in(rest)
    return Tool(
        name=rest.pop("name"),
        description=_take(rest, "description"),
        input_schema=rest.pop("input_schema"),
        cache_control=cache_control,
        extras=_extras(rest),
    )


def _tool_out(tool: Tool | Opaque) -> dict:
    if isinstance(tool, Opaque):
        return _block_out(tool)
    return _out(
        tool,
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
            "cache_control": _cache_out(tool.cache_control),
        },
    )


def _tool_choice_in(rest: dict) -> ToolChoice | None:
    if (rest.get("tool_choice") or {}).get("type") not in _TOOL_CHOICE_MODES:
        return None
    tc = dict(rest.pop("tool_choice"))
    disable = _take(tc, "disable_parallel_tool_use")
    return ToolChoice(
        mode=tc.pop("type"),
        name=_take(tc, "name"),
        parallel=None if disable is None else not disable,
        extras=_extras(tc),
    )


def _tool_choice_out(tc: ToolChoice) -> dict:
    return _out(
        tc,
        {
            "type": tc.mode,
            "name": tc.name,
            "disable_parallel_tool_use": None if tc.parallel is None else not tc.parallel,
        },
    )


def _thinking_in(rest: dict) -> ThinkingConfig | None:
    if (rest.get("thinking") or {}).get("type") not in _THINKING_MODES:
        return None
    th = dict(rest.pop("thinking"))
    return ThinkingConfig(
        mode=th.pop("type"), budget_tokens=_take(th, "budget_tokens"), extras=_extras(th)
    )


def _thinking_out(th: ThinkingConfig | None) -> dict | None:
    return None if th is None else _out(th, {"type": th.mode, "budget_tokens": th.budget_tokens})


def _request_in(body: dict) -> Request:
    rest = dict(body)
    system = _take(rest, "system")
    system_shorthand = isinstance(system, str)
    if system is not None:
        system = _system_text(_content_in(system)[0], "'system'")
    tools = _take(rest, "tools")
    return Request(
        model=rest.pop("model"),
        messages=[_message_in(m) for m in rest.pop("messages")],
        system=system,
        system_shorthand=system_shorthand,
        tools=None if tools is None else [_tool_in(t) for t in tools],
        tool_choice=_tool_choice_in(rest),
        max_tokens=_take(rest, "max_tokens"),
        temperature=_take(rest, "temperature"),
        top_p=_take(rest, "top_p"),
        top_k=_take(rest, "top_k"),
        stop=_take(rest, "stop_sequences"),
        thinking=_thinking_in(rest),
        stream=_take(rest, "stream"),
        extras=_extras(rest),
    )


def request_to_internal(body: dict) -> Request:
    return _translating("request", lambda: _request_in(body))


def request_from_internal(req: Request) -> dict:
    return _out(
        req,
        {
            "model": req.model,
            "messages": [_message_out(m) for m in req.messages],
            "system": (
                None if req.system is None else _content_out(req.system, req.system_shorthand)
            ),
            "tools": None if req.tools is None else [_tool_out(t) for t in req.tools],
            "tool_choice": None if req.tool_choice is None else _tool_choice_out(req.tool_choice),
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "top_k": req.top_k,
            "stop_sequences": req.stop,
            "thinking": _thinking_out(req.thinking),
            "stream": req.stream,
        },
    )


# --- response -----------------------------------------------------------


def usage_to_internal(raw: dict) -> Usage:
    """Anthropic's `input_tokens` excludes cache reads and writes; ours includes them."""
    rest = dict(raw)
    read = _take(rest, "cache_read_input_tokens")
    write = _take(rest, "cache_creation_input_tokens")
    uncached = _take(rest, "input_tokens")
    return Usage(
        input_tokens=None if uncached is None else uncached + (read or 0) + (write or 0),
        output_tokens=_take(rest, "output_tokens"),
        cache_read_tokens=read,
        cache_write_tokens=write,
        extras=_extras(rest),
    )


def usage_from_internal(usage: Usage) -> dict:
    total = usage.input_tokens
    read, write = usage.cache_read_tokens, usage.cache_write_tokens
    return _out(
        usage,
        {
            "input_tokens": None if total is None else total - (read or 0) - (write or 0),
            "output_tokens": usage.output_tokens,
            "cache_read_input_tokens": read,
            "cache_creation_input_tokens": write,
        },
    )


def _response_in(body: dict) -> Response:
    rest = dict(body)
    rest.pop("type", None)
    rest.pop("role", None)
    stop_reason = None
    if rest.get("stop_reason") in _KNOWN_STOP_REASONS:
        wire = rest.pop("stop_reason")
        stop_reason = _STOP_REASON_IN.get(wire, wire)
    usage = _take(rest, "usage")
    return Response(
        id=rest.pop("id"),
        model=rest.pop("model"),
        content=[_block_in(b) for b in rest.pop("content")],
        stop_reason=stop_reason,
        stop_sequence=_take(rest, "stop_sequence"),
        usage=None if usage is None else usage_to_internal(usage),
        extras=_extras(rest),
    )


def response_to_internal(body: dict) -> Response:
    return _translating("response", lambda: _response_in(body))


def stop_reason_from_internal(stop_reason: str | None) -> str | None:
    """Internal stop reason to its Anthropic wire name."""
    return None if stop_reason is None else _STOP_REASON_OUT.get(stop_reason, stop_reason)


def response_from_internal(resp: Response) -> dict:
    out = _out(
        resp,
        {
            "id": resp.id,
            "type": "message",
            "role": "assistant",
            "model": resp.model,
            "content": [_block_out(b) for b in resp.content],
            "stop_reason": stop_reason_from_internal(resp.stop_reason),
            "stop_sequence": resp.stop_sequence,
            "usage": None if resp.usage is None else usage_from_internal(resp.usage),
        },
    )
    # The API always sends both, null when unset.
    out.setdefault("stop_reason", None)
    out.setdefault("stop_sequence", None)
    return out


# --- errors -------------------------------------------------------------

# Status code to Anthropic error type, where we have no backend-specific type.
_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
    504: "timeout_error",
    529: "overloaded_error",
}
# Error types the SDK knows.
_KNOWN_ERROR_TYPES = {
    "invalid_request_error",
    "authentication_error",
    "permission_error",
    "not_found_error",
    "rate_limit_error",
    "timeout_error",
    "overloaded_error",
    "api_error",
    "billing_error",
}


def error_body(status: int, message: str, error_type: str | None = None) -> dict:
    """An Anthropic error body, for a response or an `error` stream event.
    A backend's own `error_type` wins when the SDK knows it."""
    if error_type not in _KNOWN_ERROR_TYPES:
        error_type = _ERROR_TYPES.get(status, "api_error")
    return {"type": "error", "error": {"type": error_type, "message": message}}
