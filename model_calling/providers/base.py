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
from model_calling.core import Request, Response, StreamError, StreamEvent, Tool, ToolResult
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


def uses_cache_control(request: Request) -> bool:
    """Whether any system block, message block, tool result, or tool sets `cache_control`."""
    blocks = [*(request.system or []), *(b for m in request.messages for b in m.content)]
    nested = [c for b in blocks if isinstance(b, ToolResult) for c in b.content or ()]
    tools = [t for t in request.tools or () if isinstance(t, Tool)]
    return any(getattr(x, "cache_control", None) is not None for x in (*blocks, *nested, *tools))


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
            async with self._client() as client:
                resp = await self._send(
                    client.post(
                        self.endpoint(), json=body, headers=self.headers(), timeout=self.timeout
                    )
                )
                if resp.status_code >= 400:
                    raise self._backend_error(resp.status_code, _json_or_text(resp.content))
                try:
                    native = resp.json()
                except ValueError as e:
                    raise APIResponseValidationError(
                        f"{self.name} sent a non-JSON body",
                        provider=self.name,
                        status=resp.status_code,
                        body=resp.text,
                    ) from e
            result = self._unreadable_as_backend_error(lambda: self.parse_response(native))
        except Exception as e:
            await self._notify("on_error", body, e)
            raise
        await self._notify("on_request_end", body, native)
        return result

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Yield internal events. A refused request raises before the first one.

        Every call ends with exactly one `on_request_end` or `on_error`
        callback, including when the consumer stops early (`aclose`).
        """
        body = self.build_request(request)
        body["stream"] = True
        await self._notify("on_request_start", body)
        last_chunk: Any = None
        try:
            async with self._client() as client:
                req = client.build_request(
                    "POST", self.endpoint(), json=body, headers=self.headers(), timeout=self.timeout
                )
                resp = await self._send(client.send(req, stream=True))
                try:
                    if resp.status_code >= 400:
                        raw = await resp.aread()
                        raise self._backend_error(resp.status_code, _json_or_text(raw))
                    parser = self.stream_parser()
                    async for line in self._lines(resp):
                        last_chunk = line
                        await self._notify("on_stream_chunk", body, line)
                        events = self._unreadable_as_backend_error(lambda: parser.feed(line))
                        for event in events:
                            if isinstance(event, StreamError):
                                raise _StreamFailed(event)
                            yield event
                    for event in parser.close():
                        if isinstance(event, StreamError):
                            raise _StreamFailed(event)
                        yield event
                finally:
                    await resp.aclose()
        except _StreamFailed as failed:
            # Reported to the consumer as an event, to the callbacks as an error.
            await self._notify("on_error", body, self.stream_error(failed.event))
            yield failed.event
            return
        except BaseException as e:
            await self._notify("on_error", body, e)
            raise
        await self._notify("on_request_end", body, last_chunk)

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

    async def _send(self, pending) -> httpx.Response:
        try:
            return await pending
        except httpx.TimeoutException as e:
            raise APITimeoutError(f"{self.name} timed out: {e}", provider=self.name) from e
        except httpx.TransportError as e:
            raise APIConnectionError(
                f"cannot reach {self.name} at {self.endpoint()}: {e}", provider=self.name
            ) from e

    def _backend_error(self, status: int, body: Any) -> APIError:
        message = f"{self.name} returned HTTP {status}: {self.error_message(body)}"
        logger.warning(message)
        return error_for_status(status, message, provider=self.name, body=body)

    async def _notify(self, hook: str, body: dict, *args: Any) -> None:
        """Run one callback hook for every registered callback.

        `model` and `messages` go positionally, so the rest of the body is
        passed as keywords without them.
        """
        params = {k: v for k, v in body.items() if k not in ("model", "messages")}
        for callback in self.callbacks:
            try:
                await getattr(callback, hook)(self.model, body.get("messages", []), *args, **params)
            except Exception:
                # A broken callback must not change the request's outcome.
                logger.exception("callback %s.%s failed", type(callback).__name__, hook)


class _StreamFailed(Exception):
    def __init__(self, event: StreamError):
        super().__init__(event.message)
        self.event = event


def _json_or_text(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode(errors="replace")
