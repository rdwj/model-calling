"""
OpenAI Chat Completions ingress: client requests in, responses and stream
chunks out.

Chat Completions is where the flat-content conventions live: system and
developer messages, message-level `tool_calls`, and `tool` role messages
are mapped onto blocks here. Reasoning goes out as `reasoning`, the field
OpenRouter uses; on the way in both `reasoning` and `reasoning_content` are
accepted.
"""

import json
import time
from typing import Any

from pydantic import ValidationError

from model_calling.core import (
    Base64Source,
    Image,
    Message,
    Opaque,
    Request,
    Response,
    ResponseFormat,
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
from model_calling.formats import TranslationError

FORMAT = "openai_chat"
SSE_DONE = "[DONE]"

FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}
_TOOL_CHOICE_MODES = {"auto": "auto", "none": "none", "required": "any"}
_REASONING_FIELDS = ("reasoning", "reasoning_content")
_DATA_URL_PREFIX = "data:"
_BASE64_MARKER = ";base64,"


# --- request ------------------------------------------------------------


def _parts_in(content: str | list | None) -> list:
    if content is None:
        return []
    if isinstance(content, str):
        return [Text(text=content)] if content else []
    return [_part_in(p) for p in content]


def _part_in(part: dict) -> Any:
    kind = part.get("type")
    if kind == "text":
        return Text(text=part["text"])
    if kind == "image_url":
        image_url = dict(part["image_url"])
        url = image_url.pop("url")
        extras = {FORMAT: image_url} if image_url else {}
        return Image(source=_image_source(url), extras=extras)
    return Opaque(format=FORMAT, data=part)


def _image_source(url: str) -> Base64Source | UrlSource:
    if url.startswith(_DATA_URL_PREFIX) and _BASE64_MARKER in url:
        media_type, data = url[len(_DATA_URL_PREFIX) :].split(_BASE64_MARKER, 1)
        return Base64Source(media_type=media_type, data=data)
    return UrlSource(url=url)


def _text_only(content: Any, role: str) -> list[Text]:
    blocks = _parts_in(content)
    if not all(isinstance(b, Text) for b in blocks):
        raise TranslationError(f"'{role}' message content may contain only text parts")
    return blocks


def _tool_call_in(raw: dict) -> ToolCall | Opaque:
    if raw.get("type", "function") != "function":
        return Opaque(format=FORMAT, data=raw)
    fn = raw["function"]
    arguments = fn.get("arguments") or "{}"
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return ToolCall(id=raw["id"], name=fn["name"], input=parsed)
    return ToolCall(id=raw["id"], name=fn["name"], input={}, raw_arguments=arguments)


def _assistant_in(raw: dict) -> Message:
    if raw.get("function_call") is not None:
        raise TranslationError(
            "assistant 'function_call' is deprecated and not supported; use 'tool_calls'"
        )
    blocks: list = []
    reasoning = next((raw[f] for f in _REASONING_FIELDS if raw.get(f)), None)
    if reasoning:
        blocks.append(Thinking(thinking=reasoning))
    blocks.extend(_parts_in(raw.get("content")))
    blocks.extend(_tool_call_in(tc) for tc in raw.get("tool_calls") or [])
    return Message(role="assistant", content=blocks)


def _messages_in(raw_messages: list) -> tuple[list[Text] | None, list[Message]]:
    system: list[Text] = []
    messages: list[Message] = []
    for raw in raw_messages:
        role = raw.get("role")
        if role in ("system", "developer"):
            blocks = _text_only(raw.get("content"), role)
            if messages:
                messages.append(Message(role="system", content=blocks))
            else:
                system.extend(blocks)
        elif role == "user":
            messages.append(Message(role="user", content=_parts_in(raw.get("content"))))
        elif role == "assistant":
            messages.append(_assistant_in(raw))
        elif role == "tool":
            result = ToolResult(
                tool_call_id=raw["tool_call_id"],
                content=_text_only(raw.get("content"), role),
            )
            # Consecutive tool messages answer one assistant turn: one user turn.
            prev = messages[-1] if messages else None
            if (
                prev
                and prev.role == "user"
                and all(isinstance(b, ToolResult) for b in prev.content)
            ):
                prev.content.append(result)
            else:
                messages.append(Message(role="user", content=[result]))
        else:
            raise TranslationError(f"unsupported message role {role!r}")
    return (system or None), messages


def _tools_in(raw_tools: list | None) -> list | None:
    if raw_tools is None:
        return None
    tools: list = []
    for raw in raw_tools:
        if raw.get("type") != "function":
            tools.append(Opaque(format=FORMAT, data=raw))
            continue
        fn = dict(raw["function"])
        tools.append(
            Tool(
                name=fn.pop("name"),
                description=fn.pop("description", None),
                input_schema=fn.pop("parameters", None) or {"type": "object", "properties": {}},
                extras={FORMAT: fn} if fn else {},
            )
        )
    return tools


def _tool_choice_in(raw: Any, parallel: bool | None) -> ToolChoice | None:
    if raw is None:
        return None if parallel is None else ToolChoice(mode="auto", parallel=parallel)
    if isinstance(raw, str):
        if raw not in _TOOL_CHOICE_MODES:
            raise TranslationError(f"unsupported tool_choice {raw!r}")
        return ToolChoice(mode=_TOOL_CHOICE_MODES[raw], parallel=parallel)
    if raw.get("type") == "function":
        return ToolChoice(mode="tool", name=raw["function"]["name"], parallel=parallel)
    raise TranslationError(f"unsupported tool_choice type {raw.get('type')!r}")


def _response_format_in(raw: dict | None) -> ResponseFormat | None:
    if raw is None:
        return None
    if raw.get("type") != "json_schema":
        return ResponseFormat(type=raw["type"])
    schema = raw["json_schema"]
    return ResponseFormat(
        type="json_schema",
        name=schema.get("name"),
        json_schema=schema.get("schema"),
        strict=schema.get("strict"),
    )


def _reasoning_in(effort: str | None) -> ThinkingConfig | None:
    if effort is None:
        return None
    if effort == "none":
        return ThinkingConfig(mode="disabled")
    return ThinkingConfig(mode="enabled", effort=effort)


# Deprecated function-calling fields; clients should send tools instead.
_LEGACY_FIELDS = ("functions", "function_call")


def _request_in(body: dict) -> Request:
    rest = dict(body)
    for field in _LEGACY_FIELDS:
        if field in rest:
            raise TranslationError(f"'{field}' is deprecated and not supported; use 'tools'")
    n = rest.pop("n", None)
    if n not in (None, 1):
        raise TranslationError(f"n={n} is not supported; request one choice at a time")
    system, messages = _messages_in(rest.pop("messages"))
    stop = rest.pop("stop", None)
    max_tokens = rest.pop("max_completion_tokens", None)
    legacy_max_tokens = rest.pop("max_tokens", None)
    return Request(
        model=rest.pop("model"),
        messages=messages,
        system=system,
        tools=_tools_in(rest.pop("tools", None)),
        tool_choice=_tool_choice_in(
            rest.pop("tool_choice", None), rest.pop("parallel_tool_calls", None)
        ),
        max_tokens=max_tokens if max_tokens is not None else legacy_max_tokens,
        temperature=rest.pop("temperature", None),
        top_p=rest.pop("top_p", None),
        seed=rest.pop("seed", None),
        presence_penalty=rest.pop("presence_penalty", None),
        frequency_penalty=rest.pop("frequency_penalty", None),
        stop=[stop] if isinstance(stop, str) else stop,
        thinking=_reasoning_in(rest.pop("reasoning_effort", None)),
        response_format=_response_format_in(rest.pop("response_format", None)),
        stream=rest.pop("stream", None),
        extras={FORMAT: rest} if rest else {},
    )


def request_to_internal(body: dict) -> Request:
    try:
        return _request_in(body)
    except TranslationError:
        raise
    except (KeyError, TypeError, AttributeError, ValidationError) as e:
        raise TranslationError(f"invalid Chat Completions request: {type(e).__name__}: {e}") from e


def include_usage(request: Request) -> bool:
    """Whether the client asked for a usage chunk at the end of the stream."""
    options = request.extras_for(FORMAT).get("stream_options") or {}
    return bool(options.get("include_usage"))


# --- response -----------------------------------------------------------


def usage_out(usage: Usage | None) -> dict | None:
    if usage is None:
        return None
    prompt = usage.input_tokens or 0
    completion = usage.output_tokens or 0
    out: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    if usage.cache_read_tokens is not None or usage.cache_write_tokens is not None:
        out["prompt_tokens_details"] = {
            k: v
            for k, v in (
                ("cached_tokens", usage.cache_read_tokens),
                ("cache_write_tokens", usage.cache_write_tokens),
            )
            if v is not None
        }
    if usage.reasoning_tokens is not None:
        out["completion_tokens_details"] = {"reasoning_tokens": usage.reasoning_tokens}
    return out


def _arguments(call: ToolCall) -> str:
    return call.raw_arguments if call.raw_arguments is not None else json.dumps(call.input)


def tool_call_out(call: ToolCall, index: int | None = None) -> dict:
    out = {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": ""}}
    if index is None:
        out["function"]["arguments"] = _arguments(call)
    else:
        out["index"] = index
    return out


def response_from_internal(resp: Response, model: str) -> dict:
    """`model` is the name the client asked for, which may carry a provider prefix."""
    text = "".join(b.text for b in resp.content if isinstance(b, Text))
    reasoning = "".join(b.thinking for b in resp.content if isinstance(b, Thinking))
    tool_calls = [tool_call_out(b) for b in resp.content if isinstance(b, ToolCall)]
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if reasoning:
        message["reasoning"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    elif message["content"] is None:
        message["content"] = ""
    out = {
        "id": resp.id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": FINISH_REASONS.get(resp.stop_reason, "stop"),
            }
        ],
    }
    usage = usage_out(resp.usage)
    if usage is not None:
        out["usage"] = usage
    return out


TOO_MANY_REQUESTS = 429


def error_body(status: int, message: str) -> dict:
    if status == TOO_MANY_REQUESTS:
        kind = "rate_limit_error"
    else:
        kind = "invalid_request_error" if status < 500 else "api_error"
    return {"error": {"message": message, "type": kind, "param": None, "code": None}}
