"""
The provider interface: one backend's endpoint, auth, and translation to and
from the internal format, plus the HTTP calls every provider shares.

A provider never sees an ingress format. The gateway translates the client's
request into a `Request`, hands it to the provider, and translates the
`Response` or `StreamEvent`s it gets back.
"""

import asyncio
import contextlib
import json
import logging
import threading
import uuid
import weakref
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, ClassVar

import httpx
from pydantic import ValidationError

from model_calling.callbacks import Callback
from model_calling.core import Request, Response, StreamError, StreamEvent
from model_calling.core.embeddings import EmbeddingRequest, EmbeddingResponse
from model_calling.errors import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    error_for_status,
)
from model_calling.formats import TranslationError

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 300.0
# The status a mid-stream StreamError maps to at the gateway.
BAD_GATEWAY = 502
# Long-lived streams hold their connections, so a pooled client must not cap
# them; per-call clients never did. Idle connections are still reused.
POOL_LIMITS = httpx.Limits(max_connections=None, max_keepalive_connections=20)


# Error kinds backends name inside a stream (Anthropic's and OpenAI's names).
_STREAM_ERROR_TYPES: dict[str | None, type[APIError]] = {
    "rate_limit_error": RateLimitError,
    "rate_limit_exceeded": RateLimitError,
    "overloaded_error": InternalServerError,
    "api_error": InternalServerError,
    "server_error": InternalServerError,
    "invalid_request_error": BadRequestError,
    "authentication_error": AuthenticationError,
    "permission_error": PermissionDeniedError,
    "not_found_error": NotFoundError,
}


def new_id(prefix: str = "gen-") -> str:
    """A unique id for a response or tool call the backend didn't name."""
    return f"{prefix}{uuid.uuid4().hex}"


class HttpClients:
    """One pooled `httpx.AsyncClient` per provider prefix and event loop,
    created on first use.

    A prefix's config may name a `transport` (tests inject a replay one).
    The client is built once, so a later change to `transport` needs a new
    `HttpClients`; `base_url` and `api_key` are read per request.

    An async client's connections belong to the event loop that opened them,
    so each loop gets its own clients: a script calling `asyncio.run` twice,
    or threads each running a loop, never share connections. A finished
    loop's clients go with it (they are held weakly by loop) unclosed, since
    their loop can no longer close them.
    """

    def __init__(self) -> None:
        self._by_loop: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, dict[str, httpx.AsyncClient]
        ] = weakref.WeakKeyDictionary()
        # Clients made outside any running loop (constructing a provider in
        # sync code); they bind to whichever loop first uses them.
        self._unbound: dict[str, httpx.AsyncClient] = {}
        self._lock = threading.Lock()

    def get(self, prefix: str, config: Mapping[str, Any]) -> httpx.AsyncClient:
        loop = _running_loop()
        with self._lock:
            clients = self._unbound if loop is None else self._by_loop.setdefault(loop, {})
            client = clients.get(prefix)
            if client is None or client.is_closed:
                client = clients[prefix] = httpx.AsyncClient(
                    timeout=DEFAULT_TIMEOUT, limits=POOL_LIMITS, transport=config.get("transport")
                )
            return client

    async def aclose(self) -> None:
        """Close the clients opened on the current loop."""
        loop = _running_loop()
        with self._lock:
            clients = list(self._unbound.values())
            self._unbound = {}
            if loop is not None:
                clients += self._by_loop.pop(loop, {}).values()
        for client in clients:
            await client.aclose()


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


class StreamParser(ABC):
    """Turns one backend stream, line by line, into internal events."""

    @abstractmethod
    def feed(self, line: str) -> list[StreamEvent]: ...

    def close(self) -> list[StreamEvent]:
        """Events owed once the backend closes the stream."""
        return []


class Provider(ABC):
    name: ClassVar[str]

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        callbacks: Sequence[Callback] = (),
        timeout: float = DEFAULT_TIMEOUT,
        http_client: httpx.AsyncClient | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ):
        """Without `http_client`, each call opens and closes its own client.
        `extra_headers` go on every request, after auth."""
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.callbacks = list(callbacks)
        self.timeout = timeout
        self.http_client = http_client
        self.extra_headers = dict(extra_headers or {})

    # --- what each provider defines --------------------------------------

    @abstractmethod
    def endpoint(self) -> str:
        """Full URL of the backend's chat endpoint."""

    def headers(self) -> dict[str, str]:
        auth = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return {**auth, **self.extra_headers}

    @abstractmethod
    def build_request(self, request: Request) -> dict:
        """Internal request to the backend's native body."""

    def unsupported_params(self, request: Request) -> list[str]:
        """Parameters in `request` this backend can't honor and `build_request`
        drops. The library refuses such a request unless the caller passes
        `drop_params=True`; the gateway still drops them."""
        return []

    @abstractmethod
    def parse_response(self, body: dict) -> Response:
        """Backend's native response body to an internal response."""

    @abstractmethod
    def stream_parser(self) -> StreamParser:
        """A parser for one streamed response."""

    # Embeddings are optional: a provider without them inherits these.

    def embedding_endpoint(self) -> str:
        """Full URL of the backend's embeddings endpoint."""
        raise TranslationError(f"{self.name} does not serve embeddings")

    def build_embedding_request(self, request: EmbeddingRequest) -> dict:
        """Internal embedding request to the backend's native body."""
        raise TranslationError(f"{self.name} does not serve embeddings")

    def unsupported_embedding_params(self, request: EmbeddingRequest) -> list[str]:
        """As `unsupported_params`, for embedding requests."""
        return []

    def parse_embedding_response(self, body: dict) -> EmbeddingResponse:
        """Backend's native embeddings body to an internal response."""
        raise TranslationError(f"{self.name} does not serve embeddings")

    def error_message(self, body: Any) -> str:
        """Pull the backend's own error message out of an error response body."""
        if isinstance(body, dict):
            err = body.get("error", body)
            if isinstance(err, dict):
                return str(err.get("message", err))
            return str(err)
        return str(body)

    # --- shared HTTP ------------------------------------------------------

    async def complete(self, request: Request) -> Response:
        body = self.build_request(request)
        body["stream"] = False
        await self._notify("on_request_start", body)
        try:
            native = await self._post_json(self.endpoint(), body)
            result = self._unreadable_as_backend_error(lambda: self.parse_response(native))
        except Exception as e:
            await self._notify("on_error", body, e)
            raise
        await self._notify("on_request_end", body, native)
        return result

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        body = self.build_embedding_request(request)
        url = self.embedding_endpoint()
        await self._notify_embedding("on_embedding_request_start", request, body)
        try:
            native = await self._post_json(url, body)
            result = self._unreadable_as_backend_error(
                lambda: self.parse_embedding_response(native)
            )
            if len(result.vectors) != len(request.input):
                raise APIResponseValidationError(
                    f"{self.name} returned {len(result.vectors)} embeddings "
                    f"for {len(request.input)} inputs",
                    provider=self.name,
                    body=native,
                )
        except Exception as e:
            await self._notify_embedding("on_error", request, body, e)
            raise
        await self._notify_embedding("on_embedding_request_end", request, body, native)
        return result

    async def _post_json(self, url: str, body: dict) -> Any:
        """POST `body` and return the backend's JSON answer, raising the
        typed error for a failure status or an unreadable body."""
        async with self._client() as client:
            resp = await self._send(
                client.post(url, json=body, headers=self.headers(), timeout=self.timeout), url
            )
            if resp.status_code >= 400:
                raise self._backend_error(resp.status_code, _json_or_text(resp.content))
            try:
                return resp.json()
            except ValueError as e:
                raise APIResponseValidationError(
                    f"{self.name} sent a non-JSON body",
                    provider=self.name,
                    status=resp.status_code,
                    body=resp.text,
                ) from e

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Yield internal events. A refused request raises before the first one.

        Every call ends with exactly one `on_request_end` or `on_error`
        callback. A consumer that stops early (`aclose`, or cancelling the
        task) gets `on_request_end` with `cancelled=True`. `on_stream_chunk`
        sees each internal event before it is yielded, and `on_request_end`
        gets the last one.
        """
        body = self.build_request(request)
        body["stream"] = True
        await self._notify("on_request_start", body)
        last_event: StreamEvent | None = None
        try:
            async with self._client() as client:
                req = client.build_request(
                    "POST", self.endpoint(), json=body, headers=self.headers(), timeout=self.timeout
                )
                resp = await self._send(client.send(req, stream=True), self.endpoint())
                try:
                    if resp.status_code >= 400:
                        raw = await resp.aread()
                        raise self._backend_error(resp.status_code, _json_or_text(raw))
                    parser = self.stream_parser()
                    async for line in self._lines(resp):
                        events = self._unreadable_as_backend_error(lambda: parser.feed(line))
                        for event in events:
                            last_event = await self._emit(body, event)
                            yield event
                    for event in parser.close():
                        last_event = await self._emit(body, event)
                        yield event
                finally:
                    await resp.aclose()
        except _StreamFailed as failed:
            # Reported to the consumer as an event, to the callbacks as an error.
            await self._notify("on_error", body, self.stream_error(failed.event))
            yield failed.event
            return
        except (GeneratorExit, asyncio.CancelledError):
            await _shielded(self._notify("on_request_end", body, last_event, cancelled=True))
            raise
        except BaseException as e:
            await self._notify("on_error", body, e)
            raise
        await self._notify("on_request_end", body, last_event)

    async def _emit(self, body: dict, event: StreamEvent) -> StreamEvent:
        """Run `on_stream_chunk` for an event about to be yielded; a backend
        error event instead ends the stream as a failure."""
        if isinstance(event, StreamError):
            raise _StreamFailed(event)
        await self._notify("on_stream_chunk", body, event)
        return event

    def _client(self):
        if self.http_client is not None:
            return contextlib.nullcontext(self.http_client)
        return httpx.AsyncClient(timeout=self.timeout)

    async def _lines(self, resp: httpx.Response) -> AsyncIterator[str]:
        try:
            async for line in resp.aiter_lines():
                if line.strip():
                    yield line
        except httpx.TransportError as e:
            # Read timeouts included: once the stream is up, a stall is a broken
            # connection (the gateway has always answered 502 for it).
            raise APIConnectionError(f"{self.name} stream broke: {e}", provider=self.name) from e

    def stream_error(self, event: StreamError) -> APIError:
        """The exception for a backend failure reported inside a stream. There
        is no HTTP status; the backend's error kind, when it names one,
        picks the class."""
        cls = _STREAM_ERROR_TYPES.get(event.error_type, APIError)
        return cls(event.message, provider=self.name)

    def _unreadable_as_backend_error(self, parse):
        try:
            return parse()
        except (TranslationError, ValidationError, KeyError, TypeError, AttributeError) as e:
            raise APIResponseValidationError(
                f"{self.name} sent a response we could not read: {e}", provider=self.name
            ) from e

    async def _send(self, pending, url: str) -> httpx.Response:
        try:
            return await pending
        except httpx.TimeoutException as e:
            raise APITimeoutError(f"{self.name} timed out: {e}", provider=self.name) from e
        except httpx.TransportError as e:
            raise APIConnectionError(
                f"cannot reach {self.name} at {url}: {e}", provider=self.name
            ) from e

    def _backend_error(self, status: int, body: Any) -> APIError:
        message = f"{self.name} returned HTTP {status}: {self.error_message(body)}"
        logger.warning(message)
        return error_for_status(status, message, provider=self.name, body=body)

    async def _notify(self, hook: str, body: dict, *args: Any, **extra: Any) -> None:
        """Run one callback hook for every registered callback.

        `model` and `messages` go positionally, so the rest of the body is
        passed as keywords without them, along with `extra`.
        """
        params = {k: v for k, v in body.items() if k not in ("model", "messages")}
        params.update(extra)
        for callback in self.callbacks:
            try:
                await getattr(callback, hook)(self.model, body.get("messages", []), *args, **params)
            except Exception:
                # A broken callback must not change the request's outcome.
                logger.exception("callback %s.%s failed", type(callback).__name__, hook)

    async def _notify_embedding(
        self, hook: str, request: EmbeddingRequest, body: dict, *args: Any
    ) -> None:
        """As `_notify`, with the inputs in place of messages."""
        params = {k: v for k, v in body.items() if k not in ("model", "input")}
        for callback in self.callbacks:
            try:
                await getattr(callback, hook)(self.model, request.input, *args, **params)
            except Exception:
                logger.exception("callback %s.%s failed", type(callback).__name__, hook)


# Callbacks finishing after a cancellation, held so they aren't collected.
_finishing: set[asyncio.Task] = set()


async def _shielded(coro) -> None:
    """Await `coro` in its own task, so a cancellation that keeps arriving
    (an anyio cancel scope, as Starlette uses when a client disconnects)
    can't cut it short. If the wait is cancelled, the task still finishes."""
    task = asyncio.ensure_future(coro)
    _finishing.add(task)
    task.add_done_callback(_finishing.discard)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        pass


class _StreamFailed(Exception):
    def __init__(self, event: StreamError):
        super().__init__(event.message)
        self.event = event


def _json_or_text(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode(errors="replace")
