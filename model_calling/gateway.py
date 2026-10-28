"""
Serve one ingress request through a provider: translate in, call the
backend, translate out.

Backend errors become HTTP errors with an error body in the client's format.
The library raises the backend's real status; `client_status` decides what
the gateway's own client sees.
For streams the first event is pulled before the response starts, so a
backend that refuses the request still produces a proper status code.
"""

import json
import logging
from collections.abc import AsyncIterator

from fastapi.responses import JSONResponse, StreamingResponse

from model_calling.core import StreamError, StreamEvent
from model_calling.formats import TranslationError
from model_calling.formats import chat_completions as chat
from model_calling.formats.chat_completions_stream import ChatCompletionsStreamEncoder
from model_calling.errors import APIError, APITimeoutError
from model_calling.providers import Provider

logger = logging.getLogger(__name__)

BAD_REQUEST = 400
INTERNAL_ERROR = 500
BAD_GATEWAY = 502
GATEWAY_TIMEOUT = 504
# The backend refused the gateway's credential: a gateway fault, not the
# client's, so it is reported as a bad gateway.
_CREDENTIAL_REFUSED = (401, 403)


def client_status(error: APIError) -> int:
    """The HTTP status the gateway's client sees for a backend failure."""
    if isinstance(error, APITimeoutError):
        return GATEWAY_TIMEOUT
    status = error.status
    if status is None or status < BAD_REQUEST or status >= INTERNAL_ERROR:
        return BAD_GATEWAY
    if status in _CREDENTIAL_REFUSED:
        return BAD_GATEWAY
    return status


def _sse(payload: dict | str) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {data}\n\n"


def _chat_error(status: int, message: str) -> JSONResponse:
    return JSONResponse(chat.error_body(status, message), status_code=status)


async def chat_completions(provider: Provider, body: dict):
    try:
        request = chat.request_to_internal(body)
    except TranslationError as e:
        return _chat_error(BAD_REQUEST, str(e))
    client_model = request.model
    events = None
    try:
        if request.stream:
            events = provider.stream(request)
            first = await anext(events)
            if isinstance(first, StreamError):
                await events.aclose()
                return _chat_error(first.status, first.message)
        else:
            response = await provider.complete(request)
    except TranslationError as e:
        return _chat_error(BAD_REQUEST, str(e))
    except APIError as e:
        return _chat_error(client_status(e), e.message)
    except Exception as e:
        logger.exception("request to %s failed", provider.name)
        if events is not None:
            await events.aclose()
        return _chat_error(BAD_GATEWAY, f"request to {provider.name} failed: {e}")
    if not request.stream:
        return chat.response_from_internal(response, client_model)
    encoder = ChatCompletionsStreamEncoder(client_model, chat.include_usage(request))
    return StreamingResponse(_chat_stream(encoder, first, events), media_type="text/event-stream")


async def _chat_stream(
    encoder: ChatCompletionsStreamEncoder, first: StreamEvent, rest: AsyncIterator[StreamEvent]
) -> AsyncIterator[str]:
    """SSE lines for one stream. Always closes `rest`, so the backend
    connection is released and callbacks finish even if the client leaves."""
    try:
        event = first
        while event is not None:
            for chunk in encoder.encode(event):
                yield _sse(chunk)
            if isinstance(event, StreamError):
                return
            event = await anext(rest, None)
        yield _sse(chat.SSE_DONE)
    except APIError as e:
        yield _sse(chat.error_body(client_status(e), e.message))
    except Exception as e:
        logger.exception("stream failed")
        yield _sse(chat.error_body(INTERNAL_ERROR, f"stream failed: {e}"))
    finally:
        await rest.aclose()
