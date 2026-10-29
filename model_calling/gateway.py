"""
Serve one ingress request through the library `Client`: Chat Completions,
Anthropic Messages, or Embeddings.

Clients name models by alias (`model_calling.aliases`); the gateway resolves
the alias to a backend once per request and answers with the alias in
`model`, so no backend name crosses the wire in either direction.

Backend errors become HTTP errors with an error body in the client's format.
The library raises the backend's real status; `client_status` decides what
the gateway's own client sees. A streaming call returns once the backend has
accepted the request, so a refusal still produces a proper status code.
"""

import json
import logging
import re
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse

from model_calling.aliases import (
    CONFIG_ENV,
    AliasConfig,
    AliasConfigError,
    Target,
    dimension_mismatch,
)
from model_calling.errors import (
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnsupportedParamsError,
)
from model_calling.formats import TranslationError, anthropic
from model_calling.formats import chat_completions as chat
from model_calling.library import Client
from model_calling.token_estimate import estimate_input_tokens

logger = logging.getLogger(__name__)

BAD_REQUEST = 400
NOT_FOUND = 404
INTERNAL_ERROR = 500
BAD_GATEWAY = 502
GATEWAY_TIMEOUT = 504
# The backend refused the gateway's credential: a gateway fault, not the
# client's, so it is reported as a bad gateway.
_CREDENTIAL_REFUSED = (401, 403)
# The status a typed error stands for when no backend response carried one:
# a request the library refused before sending, or an error the backend
# reported inside a stream.
_IMPLIED_STATUS = (
    (BadRequestError, BAD_REQUEST),
    (AuthenticationError, 401),
    (PermissionDeniedError, 403),
    (NotFoundError, 404),
    (RateLimitError, 429),
)
# `Client` keyword arguments a request body must not reach: they would send
# the gateway's credentials elsewhere or override its parameter policy.
RESERVED_FIELDS = ("api_base", "api_key", "drop_params")
SSE_MEDIA_TYPE = "text/event-stream"
# Stands in for a backend's name or address in what clients see.
BACKEND = "the backend"
# A URL, not counting punctuation that ends the sentence around it.
_URL = re.compile(r"[a-z][a-z0-9+.-]*://[^\s'\"<>]*[^\s'\"<>.,:;!?)]", re.IGNORECASE)

ErrorBody = Callable[[int, str], dict]


def client_status(error: APIError) -> int:
    """The HTTP status the gateway's client sees for a failure."""
    if isinstance(error, APITimeoutError):
        return GATEWAY_TIMEOUT
    status = error.status
    if status is None:
        status = next((s for cls, s in _IMPLIED_STATUS if isinstance(error, cls)), None)
    if status is None or status < BAD_REQUEST or status >= INTERNAL_ERROR:
        return BAD_GATEWAY
    if status in _CREDENTIAL_REFUSED:
        return BAD_GATEWAY
    return status


# --- the three ingresses ------------------------------------------------------


async def chat_completions(client: Client, body: Any, aliases: AliasConfig, *, drop_params: bool):
    async def call(target: Target, params):
        messages = params.pop("messages", None)
        return await client.acompletion(
            target.model, messages, drop_params=drop_params, **_backend(target), **params
        )

    return await _serve(body, aliases, call, chat.error_body, _sse, _sse(chat.SSE_DONE))


async def messages(client: Client, body: Any, aliases: AliasConfig, *, drop_params: bool):
    async def call(target: Target, params):
        messages, max_tokens = params.pop("messages", None), params.pop("max_tokens", None)
        return await client.amessages(
            target.model,
            messages,
            max_tokens=max_tokens,
            drop_params=drop_params,
            **_backend(target),
            **params,
        )

    return await _serve(body, aliases, call, anthropic.error_body, _anthropic_event, None)


async def embeddings(client: Client, body: Any, aliases: AliasConfig, *, drop_params: bool):
    async def call(target: Target, params):
        return await client.aembedding(
            target.model,
            params.pop("input", None),
            drop_params=drop_params,
            **_backend(target),
            **params,
        )

    return await _serve(
        body, aliases, call, chat.error_body, encode=None, end=None, check=dimension_mismatch
    )


def count_tokens(body: Any, aliases: AliasConfig):
    """`POST /v1/messages/count_tokens`, answered locally with an estimate.

    The body is validated as `/v1/messages` validates it, without a backend call.
    """

    def error(status: int, message: str) -> JSONResponse:
        return JSONResponse(anthropic.error_body(status, message), status_code=status)

    if not isinstance(body, dict):
        return error(BAD_REQUEST, "the request body must be a JSON object")
    model = body.get("model")
    if not isinstance(model, str) or not model:
        return error(BAD_REQUEST, "model is required")
    if model not in aliases.models:
        return error(NOT_FOUND, unknown_model(model, aliases))
    try:
        request = anthropic.request_to_internal(body)
    except TranslationError as e:
        return error(BAD_REQUEST, str(e))
    return {"input_tokens": estimate_input_tokens(request)}


def unknown_model(model: str, aliases: AliasConfig) -> str:
    if not aliases.models:
        return f"model {model!r} not found: no models are configured (set {CONFIG_ENV})"
    return f"model {model!r} not found; this gateway serves {', '.join(aliases.aliases)}"


# --- plumbing -----------------------------------------------------------------


async def _serve(
    body, aliases: AliasConfig, call, error_body: ErrorBody, encode, end: str | None, check=None
):
    """Validate the envelope, resolve the alias, make the call, and answer
    in the ingress's format.

    `encode` turns one streamed item into SSE text, or is None for an
    ingress that never streams; `end` closes a stream. `check` may refuse a
    response: it returns an error message, or None.
    """

    def error(status: int, message: str) -> JSONResponse:
        return JSONResponse(error_body(status, message), status_code=status)

    if not isinstance(body, dict):
        return error(BAD_REQUEST, "the request body must be a JSON object")
    params = dict(body)
    model = params.pop("model", None)
    if not isinstance(model, str) or not model:
        return error(BAD_REQUEST, "model is required")
    reserved = sorted(k for k in RESERVED_FIELDS if k in params)
    if reserved:
        return error(BAD_REQUEST, f"the gateway does not accept {', '.join(reserved)}")
    try:
        target = aliases.resolve(model)
    except AliasConfigError as e:
        logger.error("model %r is misconfigured: %s", model, e)
        return error(BAD_GATEWAY, f"model {model!r} is misconfigured on the gateway")
    if target is None:
        return error(NOT_FOUND, unknown_model(model, aliases))
    try:
        result = await call(target, params)
    except UnsupportedParamsError as e:
        # Strict mode. The library's hint (drop_params) isn't the gateway client's to use.
        return error(BAD_REQUEST, f"model {model!r} cannot honor {', '.join(e.params)}")
    except APIError as e:
        return error(client_status(e), public_message(e.message, target))
    except Exception:
        logger.exception("request for %r to %s failed", model, target.provider)
        return error(BAD_GATEWAY, f"request for {model!r} failed")
    if encode is None or not params.get("stream"):
        response = _with_alias(result.to_dict(), model)
        problem = check(response, target) if check else None
        if problem:
            logger.error(problem)
            return error(BAD_GATEWAY, problem)
        return response

    def encode_with_alias(payload: dict) -> str:
        return encode(_with_alias(payload, model))

    def stream_error(status: int, message: str) -> dict:
        return error_body(status, public_message(message, target))

    return StreamingResponse(
        _stream(result, encode_with_alias, stream_error, end), media_type=SSE_MEDIA_TYPE
    )


def public_message(message: str, target: Target) -> str:
    """An error message fit for the gateway's client: the backend's name,
    its model name, and any address it mentions are replaced, since which
    backend serves an alias is the gateway's business. The full message is
    logged where the library raised it."""
    message = _URL.sub(BACKEND, message)
    native = target.model.split("/", 1)[1]
    for name in sorted({target.model, native}, key=len, reverse=True):
        message = re.sub(rf"(?<![\w.:/-]){re.escape(name)}(?![\w.:/-])", target.alias, message)
    return re.sub(rf"\b{re.escape(target.provider)}\b", BACKEND, message, flags=re.IGNORECASE)


def _backend(target: Target) -> dict:
    """The per-call overrides a target's backend sets."""
    return {k: v for k, v in (("api_base", target.api_base), ("api_key", target.api_key)) if v}


def _with_alias(payload: dict, alias: str) -> dict:
    """Show the alias, not the backend's model, wherever a response names it:
    the body or chunk's `model`, or a Messages stream's `message_start`."""
    if "model" in payload:
        payload["model"] = alias
    message = payload.get("message")
    if isinstance(message, dict) and "model" in message:
        message["model"] = alias
    return payload


async def _stream(
    items: AsyncIterator[Any], encode, error_body: ErrorBody, end: str | None
) -> AsyncIterator[str]:
    """SSE text for one stream. Always closes `items`, so the backend
    connection is released and callbacks finish even if the client leaves."""
    try:
        async for item in items:
            yield encode(item.to_dict())
        if end is not None:
            yield end
    except APIError as e:
        yield encode(error_body(client_status(e), e.message))
    except Exception:
        logger.exception("stream failed")
        yield encode(error_body(INTERNAL_ERROR, "stream failed"))
    finally:
        await items.aclose()


def _sse(payload: dict | str) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {data}\n\n"


def _anthropic_event(payload: dict) -> str:
    """Anthropic names each event in an `event:` line, as its `type`."""
    return f"event: {payload['type']}\n{_sse(payload)}"
