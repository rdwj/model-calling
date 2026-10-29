"""
The call steps every `Client` method shares: translate the request, call
or stream from the provider, and type what comes back, with failures
mapped onto the library's exceptions.
"""

from collections.abc import AsyncIterator, Callable
from typing import Any, TypeVar

from pydantic import ValidationError

from model_calling.core import Request, Response, StreamError, StreamEvent
from model_calling.errors import APIResponseValidationError, BadRequestError
from model_calling.formats import TranslationError
from model_calling.providers import Provider

T = TypeVar("T")


def translate(provider: Provider, fn: Callable[[], Request]) -> Request:
    try:
        return fn()
    except TranslationError as e:
        raise BadRequestError(str(e), provider=provider.name) from e


async def complete(provider: Provider, request: Request) -> Response:
    """`provider.complete`, with a request the backend can't express as a
    `BadRequestError`. Unreadable responses are already typed by the provider."""
    try:
        return await provider.complete(request)
    except TranslationError as e:
        raise BadRequestError(str(e), provider=provider.name) from e


async def open_stream(provider: Provider, request: Request) -> AsyncIterator[StreamEvent]:
    """Start a stream and read its first event, so a refused request raises
    before the caller starts iterating."""
    events = provider.stream(request)
    try:
        first = await anext(events)
    except TranslationError as e:
        raise BadRequestError(str(e), provider=provider.name) from e
    except StopAsyncIteration:
        raise APIResponseValidationError(
            f"{provider.name} ended the stream before it started", provider=provider.name
        ) from None
    except BaseException:
        await events.aclose()
        raise
    if isinstance(first, StreamError):
        await events.aclose()
        raise provider.stream_error(first)
    return _from(first, events, provider)


async def _from(
    first: StreamEvent, rest: AsyncIterator[StreamEvent], provider: Provider
) -> AsyncIterator[StreamEvent]:
    """`first` then `rest`, raising on a backend error event. Always closes
    `rest`, so the connection is released and callbacks finish."""
    try:
        event: StreamEvent | None = first
        while event is not None:
            if isinstance(event, StreamError):
                raise provider.stream_error(event)
            yield event
            event = await anext(rest, None)
    finally:
        await rest.aclose()


def typed(provider: Provider, build: Callable[[], T]) -> T:
    """Our output as a typed object. A failure means the backend sent
    something our formats can't express, so it is typed as such."""
    try:
        return build()
    except (ValidationError, TranslationError, RuntimeError) as e:
        raise APIResponseValidationError(
            f"{provider.name} sent a response we could not express: {e}", provider=provider.name
        ) from e


async def encoded(
    provider: Provider,
    events: AsyncIterator[StreamEvent],
    encode: Callable[[StreamEvent], list[dict]],
    parse: Callable[[dict], Any],
) -> AsyncIterator[Any]:
    try:
        async for event in events:
            for item in typed(provider, lambda: encode(event)):
                yield typed(provider, lambda: parse(item))
    finally:
        await events.aclose()
