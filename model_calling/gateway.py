"""
Serve one ingress request through the library `Client`: Chat Completions,
Anthropic Messages, or Embeddings.

Backend errors become HTTP errors with an error body in the client's format.
The library raises the backend's real status; `client_status` decides what
the gateway's own client sees. A streaming call returns once the backend has
accepted the request, so a refusal still produces a proper status code.
"""

import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse

from model_calling.errors import APIError, APITimeoutError, BadRequestError, UnsupportedParamsError
from model_calling.formats import anthropic
from model_calling.formats import chat_completions as chat
from model_calling.library import Client
from model_calling.providers import split_model

logger = logging.getLogger(__name__)

BAD_REQUEST = 400
INTERNAL_ERROR = 500
BAD_GATEWAY = 502
GATEWAY_TIMEOUT = 504
# The backend refused the gateway's credential: a gateway fault, not the
# client's, so it is reported as a bad gateway.
_CREDENTIAL_REFUSED = (401, 403)
# `Client` keyword arguments a request body must not reach: they would send
# the gateway's credentials elsewhere or override its parameter policy.
RESERVED_FIELDS = ("api_base", "api_key", "drop_params")
SSE_MEDIA_TYPE = "text/event-stream"

ErrorBody = Callable[[int, str], dict]


def client_status(error: APIError) -> int:
    """The HTTP status the gateway's client sees for a failure."""
    if isinstance(error, APITimeoutError):
        return GATEWAY_TIMEOUT
    status = error.status
    if status is None and isinstance(error, BadRequestError):
        # The library refused the request without a backend response.
        return BAD_REQUEST
    if status is None or status < BAD_REQUEST or status >= INTERNAL_ERROR:
        return BAD_GATEWAY
    if status in _CREDENTIAL_REFUSED:
        return BAD_GATEWAY
    return status


# --- the three ingresses ------------------------------------------------------


async def chat_completions(client: Client, body: Any, *, drop_params: bool):
    async def call(model, params):
        messages = params.pop("messages", None)
        return await client.acompletion(model, messages, drop_params=drop_params, **params)

    return await _serve(body, call, chat.error_body, _sse, _sse(chat.SSE_DONE))


async def messages(client: Client, body: Any, *, drop_params: bool):
    async def call(model, params):
        messages, max_tokens = params.pop("messages", None), params.pop("max_tokens", None)
        return await client.amessages(
            model, messages, max_tokens=max_tokens, drop_params=drop_params, **params
        )

    return await _serve(body, call, anthropic.error_body, _anthropic_event, None)


async def embeddings(client: Client, body: Any, *, drop_params: bool):
    async def call(model, params):
        return await client.aembedding(
            model, params.pop("input", None), drop_params=drop_params, **params
        )

    return await _serve(body, call, chat.error_body, encode=None, end=None)


# --- plumbing -----------------------------------------------------------------


async def _serve(body, call, error_body: ErrorBody, encode, end: str | None):
    """Validate the envelope, make the call, and answer in the ingress's format.

    `encode` turns one streamed item into SSE text, or is None for an
    ingress that never streams; `end` closes a stream.
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
        result = await call(model, params)
    except UnsupportedParamsError as e:
        # Strict mode. The library's hint (drop_params) isn't the gateway client's to use.
        return error(BAD_REQUEST, f"{e.provider} cannot honor {', '.join(e.params)}")
    except APIError as e:
        return error(client_status(e), e.message)
    except Exception as e:
        provider = split_model(model)[0]
        logger.exception("request to %s failed", provider)
        return error(BAD_GATEWAY, f"request to {provider} failed: {e}")
    if encode is None or not params.get("stream"):
        return result.to_dict()
    return StreamingResponse(_stream(result, encode, error_body, end), media_type=SSE_MEDIA_TYPE)


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
    except Exception as e:
        logger.exception("stream failed")
        yield encode(error_body(INTERNAL_ERROR, f"stream failed: {e}"))
    finally:
        await items.aclose()


def _sse(payload: dict | str) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {data}\n\n"


def _anthropic_event(payload: dict) -> str:
    """Anthropic names each event in an `event:` line, as its `type`."""
    return f"event: {payload['type']}\n{_sse(payload)}"
