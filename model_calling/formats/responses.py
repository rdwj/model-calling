"""
OpenAI Responses ingress, stateless: client requests in, responses out.
Stream events are in `responses_stream`.

Nothing is stored. `store` is accepted (absent, true, or false) and every
response reports `store: false`; a request that needs stored state
(`previous_response_id`, `conversation`, `background`, a stored `prompt`)
is refused naming the field. The first cut covers input message items
(text and images), `function_call` and `function_call_output` items, and
reasoning items passed back; function tools only. Built-in tools,
encrypted reasoning, and other item and tool types are refused naming
them, rather than dropped.

`instructions` becomes the system prompt, ahead of any leading system or
developer input messages. Each output block becomes one output item: text
a message item, a tool call a `function_call` item, and thinking a
reasoning item carrying it as summary text. Item ids derive from the
response id, so a response and its stream name items alike.
"""

import json
import time
from typing import Any

from pydantic import ValidationError

from model_calling.core import (
    FileSource,
    Image,
    Message,
    Opaque,
    RedactedThinking,
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
)
from model_calling.formats import TranslationError
from model_calling.formats.chat_completions import error_body, image_source, tool_call_in

FORMAT = "openai_responses"
__all__ = ["FORMAT", "error_body", "request_to_internal", "response_from_internal"]

# Fields that need state this gateway doesn't keep. A null (or false) is no request for it.
STATEFUL_FIELDS = ("previous_response_id", "conversation", "background", "prompt")
ENCRYPTED_REASONING = "reasoning.encrypted_content"
# OpenAI's built-in tools, by every `type` the SDK names them with.
BUILT_IN_TOOLS = {
    "web_search": "web_search",
    "web_search_2025_08_26": "web_search",
    "web_search_preview": "web_search",
    "web_search_preview_2025_03_11": "web_search",
    "file_search": "file_search",
    "code_interpreter": "code_interpreter",
    "computer": "computer use",
    "computer_use_preview": "computer use",
}
_TOOL_CHOICE_MODES = {"auto": "auto", "none": "none", "required": "any"}
_TOOL_CHOICE_OUT = {v: k for k, v in _TOOL_CHOICE_MODES.items()}
_SYSTEM_ROLES = ("system", "developer")
_TEXT_PARTS = ("input_text", "output_text")
# Internal stop reasons a response doesn't complete on -> incomplete_details.reason.
INCOMPLETE_REASONS = {
    "max_tokens": "max_output_tokens",
    "context_window_exceeded": "max_output_tokens",
    "refusal": "content_filter",
}
EMPTY_SCHEMA = {"type": "object", "properties": {}}
# Output item id prefixes, as OpenAI uses them.
ITEM_PREFIXES = {"message": "msg", "function_call": "fc", "reasoning": "rs"}


def _extras(rest: dict) -> dict:
    return {FORMAT: rest} if rest else {}


# --- request ------------------------------------------------------------


def _refuse_stateful(body: dict) -> None:
    for field in STATEFUL_FIELDS:
        if body.get(field) not in (None, False):
            raise TranslationError(f"{field} is not supported: this gateway keeps no state")
    if ENCRYPTED_REASONING in (body.get("include") or ()):
        raise TranslationError(
            f"include {ENCRYPTED_REASONING} is not supported: reasoning is returned as summary text"
        )


def _text(part: dict) -> Text:
    rest = dict(part)
    del rest["type"]
    return Text(text=rest.pop("text"), extras=_extras(rest))


def _image(part: dict) -> Image:
    rest = dict(part)
    del rest["type"]
    url, file_id = rest.pop("image_url", None), rest.pop("file_id", None)
    if url:
        source = image_source(url)
    elif file_id:
        source = FileSource(file_id=file_id)
    else:
        raise TranslationError("an input_image needs an image_url or a file_id")
    return Image(source=source, extras=_extras(rest))


def _part_in(part: dict, allowed: tuple) -> Text | Image:
    kind = part.get("type")
    if kind not in allowed:
        raise TranslationError(f"content part type {kind!r} is not supported here")
    return _image(part) if kind == "input_image" else _text(part)


def _content_in(content: str | list, allowed: tuple) -> tuple[list, bool]:
    if isinstance(content, str):
        return ([Text(text=content)] if content else []), True
    return [_part_in(p, allowed) for p in content], False


def _turn(messages: list[Message], role: str) -> Message:
    """The turn a non-message item belongs to: the last one if it has
    `role`, else a new one."""
    if not messages or messages[-1].role != role:
        messages.append(Message(role=role, content=[]))
    return messages[-1]


def _message_in(item: dict, system: list[Text], messages: list[Message]) -> None:
    rest = dict(item)
    rest.pop("type", None)
    role = rest.pop("role")
    if role in _SYSTEM_ROLES:
        blocks, _ = _content_in(rest.pop("content"), ("input_text",))
        if messages:
            messages.append(Message(role="system", content=blocks, extras=_extras(rest)))
        else:
            system.extend(blocks)
        return
    if role == "user":
        allowed = ("input_text", "input_image")
    elif role == "assistant":
        allowed = _TEXT_PARTS
    else:
        raise TranslationError(f"unsupported message role {role!r}")
    blocks, shorthand = _content_in(rest.pop("content"), allowed)
    messages.append(
        Message(role=role, content=blocks, content_shorthand=shorthand, extras=_extras(rest))
    )


def _function_call_in(item: dict) -> ToolCall | Opaque:
    rest = dict(item)
    for key in ("type", "call_id", "name", "arguments"):
        rest.pop(key)
    call = tool_call_in(
        {"id": item["call_id"], "function": {"name": item["name"], "arguments": item["arguments"]}}
    )
    call.extras = _extras(rest)
    return call


def _function_call_output_in(item: dict) -> ToolResult:
    rest = dict(item)
    del rest["type"]
    call_id = rest.pop("call_id", None)
    if not call_id:
        raise TranslationError("a function_call_output item needs a call_id")
    content, shorthand = _content_in(rest.pop("output"), ("input_text", "input_image"))
    return ToolResult(
        tool_call_id=call_id, content=content, content_shorthand=shorthand, extras=_extras(rest)
    )


def _reasoning_in(item: dict) -> list[Thinking]:
    if item.get("encrypted_content") is not None:
        raise TranslationError(f"{ENCRYPTED_REASONING} is not supported in input reasoning items")
    return [Thinking(thinking=part["text"]) for part in item["summary"]]


def _input_in(raw: str | list, system: list[Text]) -> list[Message]:
    if isinstance(raw, str):
        return [Message(role="user", content=[Text(text=raw)], content_shorthand=True)]
    messages: list[Message] = []
    for item in raw:
        kind = item.get("type", "message")
        if kind == "message":
            _message_in(item, system, messages)
        elif kind == "function_call":
            _turn(messages, "assistant").content.append(_function_call_in(item))
        elif kind == "reasoning":
            _turn(messages, "assistant").content.extend(_reasoning_in(item))
        elif kind == "function_call_output":
            # Consecutive outputs answer one assistant turn: one user turn.
            turn = messages[-1] if messages else None
            if not (
                turn
                and turn.role == "user"
                and all(isinstance(b, ToolResult) for b in turn.content)
            ):
                turn = Message(role="user", content=[])
                messages.append(turn)
            turn.content.append(_function_call_output_in(item))
        else:
            raise TranslationError(f"input item type {kind!r} is not supported")
    return messages


def _tool_in(raw: dict) -> Tool:
    kind = raw.get("type")
    if kind in BUILT_IN_TOOLS:
        raise TranslationError(
            f"built-in tool {kind!r} ({BUILT_IN_TOOLS[kind]}) is not supported: "
            "this gateway runs function tools only"
        )
    if kind != "function":
        raise TranslationError(f"tool type {kind!r} is not supported: use function tools")
    rest = dict(raw)
    del rest["type"]
    return Tool(
        name=rest.pop("name"),
        description=rest.pop("description", None),
        input_schema=rest.pop("parameters", None) or dict(EMPTY_SCHEMA),
        # Null leaves strictness to the backend, as an absent field does.
        strict=rest.pop("strict", None),
        extras=_extras(rest),
    )


def _tool_choice_in(raw: Any, parallel: bool | None) -> ToolChoice | None:
    if raw is None:
        return None if parallel is None else ToolChoice(mode="auto", parallel=parallel)
    if isinstance(raw, str):
        if raw not in _TOOL_CHOICE_MODES:
            raise TranslationError(f"unsupported tool_choice {raw!r}")
        return ToolChoice(mode=_TOOL_CHOICE_MODES[raw], parallel=parallel)
    if raw.get("type") == "function":
        return ToolChoice(mode="tool", name=raw["name"], parallel=parallel)
    raise TranslationError(f"tool_choice type {raw.get('type')!r} is not supported")


def _text_format_in(rest: dict) -> ResponseFormat | None:
    text = rest.get("text")
    if not isinstance(text, dict) or text.get("format") is None:
        return None
    text = dict(rest.pop("text"))
    fmt = dict(text.pop("format"))
    if text:
        rest["text"] = text
    kind = fmt.pop("type")
    return ResponseFormat(
        type=kind,
        name=fmt.pop("name", None),
        json_schema=fmt.pop("schema", None),
        strict=fmt.pop("strict", None),
        extras=_extras(fmt),
    )


def _thinking_in(rest: dict) -> ThinkingConfig | None:
    reasoning = rest.get("reasoning")
    if not isinstance(reasoning, dict) or reasoning.get("effort") is None:
        return None
    reasoning = dict(rest.pop("reasoning"))
    effort = reasoning.pop("effort")
    if effort == "none":
        return ThinkingConfig(mode="disabled", extras=_extras(reasoning))
    return ThinkingConfig(mode="enabled", effort=effort, extras=_extras(reasoning))


def _request_in(body: dict) -> Request:
    _refuse_stateful(body)
    rest = dict(body)
    for field in (*STATEFUL_FIELDS, "store"):
        rest.pop(field, None)
    instructions = rest.pop("instructions", None)
    if instructions is not None and not isinstance(instructions, str):
        raise TranslationError("instructions must be a string")
    system = [Text(text=instructions)] if instructions else []
    raw_input = rest.pop("input", None)
    if raw_input is None:
        raise TranslationError("input is required")
    messages = _input_in(raw_input, system)
    tools = rest.pop("tools", None)
    return Request(
        model=rest.pop("model"),
        messages=messages,
        system=system or None,
        # The system prompt starts with `instructions`, a bare string.
        system_shorthand=bool(instructions),
        tools=None if tools is None else [_tool_in(t) for t in tools],
        tool_choice=_tool_choice_in(
            rest.pop("tool_choice", None), rest.pop("parallel_tool_calls", None)
        ),
        max_tokens=rest.pop("max_output_tokens", None),
        temperature=rest.pop("temperature", None),
        top_p=rest.pop("top_p", None),
        thinking=_thinking_in(rest),
        response_format=_text_format_in(rest),
        stream=rest.pop("stream", None),
        extras=_extras(rest),
    )


def request_to_internal(body: dict) -> Request:
    try:
        return _request_in(body)
    except TranslationError:
        raise
    except (KeyError, TypeError, AttributeError, ValidationError) as e:
        raise TranslationError(f"invalid Responses request: {type(e).__name__}: {e}") from e


# --- response -----------------------------------------------------------


def item_id(kind: str, response_id: str, output_index: int) -> str:
    return f"{ITEM_PREFIXES[kind]}_{response_id}_{output_index}"


def output_text(text: str) -> dict:
    return {"type": "output_text", "text": text, "annotations": [], "logprobs": []}


def summary_text(text: str) -> dict:
    return {"type": "summary_text", "text": text}


def arguments(call: ToolCall) -> str:
    return call.raw_arguments if call.raw_arguments is not None else json.dumps(call.input)


def output_item(block: Any, item: str, status: str) -> dict | None:
    """One internal block as an output item with id `item`; None for one
    with no Responses form that is safe to leave out (redacted thinking)."""
    if isinstance(block, Text):
        return {
            "id": item,
            "type": "message",
            "role": "assistant",
            "status": status,
            "content": [output_text(block.text)],
        }
    if isinstance(block, ToolCall):
        return {
            "id": item,
            "type": "function_call",
            "status": status,
            "arguments": arguments(block),
            "call_id": block.id,
            "name": block.name,
        }
    if isinstance(block, Thinking):
        return {
            "id": item,
            "type": "reasoning",
            "status": status,
            "summary": [summary_text(block.thinking)],
        }
    if isinstance(block, RedactedThinking):
        return None
    kind = block.data.get("type") if isinstance(block, Opaque) else block.type
    raise TranslationError(f"no Responses output item for a {kind!r} block")


def item_kind(block: Any) -> str | None:
    """The output item type `block` becomes, or None if it is left out."""
    return {Text: "message", ToolCall: "function_call", Thinking: "reasoning"}.get(type(block))


def output_items(response_id: str, blocks: list, status: str) -> list[dict]:
    """The output items for `blocks`. In a response that didn't complete,
    the last item is the one cut short."""
    items = []
    for block in blocks:
        kind = item_kind(block)
        item = output_item(block, kind and item_id(kind, response_id, len(items)), "completed")
        if item is not None:
            items.append(item)
    if status == "incomplete" and items:
        items[-1]["status"] = "incomplete"
    return items


def status_for(stop_reason: str | None) -> str:
    return "incomplete" if stop_reason in INCOMPLETE_REASONS else "completed"


def usage_out(usage: Usage | None) -> dict:
    """Responses usage. Every count is required, so unknown ones are 0."""
    usage = usage or Usage()
    inputs, outputs = usage.input_tokens or 0, usage.output_tokens or 0
    return {
        "input_tokens": inputs,
        "input_tokens_details": {
            "cache_write_tokens": usage.cache_write_tokens or 0,
            "cached_tokens": usage.cache_read_tokens or 0,
        },
        "output_tokens": outputs,
        "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens or 0},
        "total_tokens": inputs + outputs,
    }


def _tool_out(tool: Tool) -> dict:
    return {
        "name": tool.name,
        "parameters": tool.input_schema,
        "strict": tool.strict,
        "type": "function",
        **tool.extras_for(FORMAT),
        **({} if tool.description is None else {"description": tool.description}),
    }


def _tool_choice_out(choice: ToolChoice | None) -> str | dict:
    if choice is None:
        return "auto"
    if choice.mode == "tool":
        return {"type": "function", "name": choice.name}
    return _TOOL_CHOICE_OUT[choice.mode]


def _text_out(fmt: ResponseFormat | None) -> dict:
    if fmt is None or fmt.type != "json_schema":
        return {"format": {"type": "text" if fmt is None else fmt.type}}
    return {
        "format": {
            "type": "json_schema",
            "name": fmt.name or "response",
            "schema": fmt.json_schema or {},
            "strict": fmt.strict,
        }
    }


def _effort_out(thinking: ThinkingConfig) -> str | None:
    return "none" if thinking.mode == "disabled" else thinking.effort


def response_body(
    response_id: str,
    model: str,
    request: Request,
    *,
    status: str,
    created_at: int,
    output: list[dict] | None = None,
    stop_reason: str | None = None,
    usage: Usage | None = None,
) -> dict:
    """A Responses `response` object, echoing the request's settings as the
    API does. `usage` is reported once the response is done."""
    choice = request.tool_choice
    thinking = request.thinking
    reason = INCOMPLETE_REASONS.get(stop_reason) if status == "incomplete" else None
    return {
        "id": response_id,
        "created_at": created_at,
        "error": None,
        "incomplete_details": None if reason is None else {"reason": reason},
        "instructions": request.system[0].text if request.system_shorthand else None,
        "model": model,
        "object": "response",
        "output": output or [],
        "parallel_tool_calls": (
            True if choice is None or choice.parallel is None else choice.parallel
        ),
        "temperature": request.temperature,
        "tool_choice": _tool_choice_out(choice),
        "tools": [_tool_out(t) for t in request.tools or () if isinstance(t, Tool)],
        "top_p": request.top_p,
        "max_output_tokens": request.max_tokens,
        "reasoning": None if thinking is None else {"effort": _effort_out(thinking)},
        "status": status,
        "text": _text_out(request.response_format),
        "usage": None if status == "in_progress" else usage_out(usage),
        # Nothing is stored, whatever the request asked.
        "store": False,
    }


def response_from_internal(resp: Response, model: str, request: Request) -> dict:
    """`model` is the name the client asked for; `request` is the internal
    form of the client's request, whose settings the response echoes."""
    status = status_for(resp.stop_reason)
    return response_body(
        resp.id,
        model,
        request,
        status=status,
        created_at=int(time.time()),
        output=output_items(resp.id, resp.content, status),
        stop_reason=resp.stop_reason,
        usage=resp.usage,
    )


def stream_error_event(status: int, message: str, sequence_number: int) -> dict:
    """An `error` stream event for a backend that failed mid-stream."""
    code = error_body(status, message)["error"]["type"]
    return {
        "type": "error",
        "code": code,
        "message": message,
        "param": None,
        "sequence_number": sequence_number,
    }
