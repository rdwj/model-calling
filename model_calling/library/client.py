"""
The in-process library API: `Client`, with Chat Completions and Anthropic
Messages calls side by side.

A client owns its per-prefix config, its callbacks, and pooled HTTP
clients. Async calls pool on the caller's event loop; sync calls run on a
private loop thread (`_sync.LoopThread`) with a pool of their own. Close a
client with `aclose()` / `async with` after async use and `close()` /
`with` after sync use.
"""

import copy
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from typing import Any

from model_calling.callbacks import Callback
from model_calling.core import Request, StreamError, StreamEvent, Usage
from model_calling.errors import APIResponseValidationError, BadRequestError
from model_calling.formats import TranslationError
from model_calling.formats import anthropic
from model_calling.formats import chat_completions as chat
from model_calling.formats.anthropic_stream import AnthropicStreamEncoder
from model_calling.formats.chat_completions_stream import ChatCompletionsStreamEncoder
from model_calling.library._sync import LoopThread
from model_calling.providers import (
    PROVIDERS,
    HttpClients,
    Provider,
    default_configs,
    provider_for,
    split_model,
)
from model_calling.types.chat import ChatCompletion, ChatCompletionChunk
from model_calling.types.messages import Message, MessageStreamEvent, parse_stream_event


class Client:
    def __init__(
        self,
        *,
        config: Mapping[str, Mapping[str, Any]] | None = None,
        callbacks: Sequence[Callback] = (),
    ):
        """`config` maps a model prefix (`ollama`, `vllm`, `openai`,
        `openrouter`) to settings merged over `providers.default_configs()`:
        `base_url`, `api_key`, `headers`, `callbacks`, and `transport` (an
        httpx transport, mostly for tests). `callbacks` run for every prefix,
        before the prefix's own.
        """
        self.config = default_configs()
        for prefix, overrides in (config or {}).items():
            self.config[prefix] = {**self.config.get(prefix, {}), **overrides}
        self.callbacks = list(callbacks)
        self._http = HttpClients()
        self._sync_http = HttpClients()
        self._sync: LoopThread | None = None
        self._sync_lock = threading.Lock()

    # --- Chat Completions ---------------------------------------------------

    async def acompletion(
        self,
        model: str,
        messages: list[dict],
        *,
        stream: bool = False,
        api_base: str | None = None,
        api_key: str | None = None,
        **params: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """A Chat Completions call. `model` is `prefix/name`, as `ollama/qwen3:4b`.

        With `stream=True` this returns an async iterator of chunks, once the
        backend has accepted the request; a refusal raises here, not mid-loop.
        Other keyword arguments are Chat Completions request fields.
        """
        return await self._completion(
            self._http, model, messages, stream, api_base, api_key, params
        )

    def completion(
        self,
        model: str,
        messages: list[dict],
        *,
        stream: bool = False,
        api_base: str | None = None,
        api_key: str | None = None,
        **params: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """The sync form of `acompletion`; streams are sync iterators."""
        return self._run_sync(
            self._completion(self._sync_http, model, messages, stream, api_base, api_key, params),
            stream,
        )

    async def _completion(self, http, model, messages, stream, api_base, api_key, params):
        body = {**params, "model": model, "messages": messages, "stream": stream}
        provider = self._provider(http, model, api_base, api_key)
        request = _translate(provider, lambda: chat.request_to_internal(body))
        if not stream:
            response = await provider.complete(request)
            return ChatCompletion.model_validate(chat.response_from_internal(response, model))
        encoder = ChatCompletionsStreamEncoder(model, chat.include_usage(request))
        events = await _open_stream(provider, request)
        return _encoded(events, encoder.encode, ChatCompletionChunk.model_validate)

    # --- Anthropic Messages -------------------------------------------------

    async def amessages(
        self,
        model: str,
        messages: list[dict],
        *,
        max_tokens: int,
        stream: bool = False,
        api_base: str | None = None,
        api_key: str | None = None,
        **params: Any,
    ) -> Message | AsyncIterator[MessageStreamEvent]:
        """An Anthropic Messages call over any backend. `model` is `prefix/name`.

        With `stream=True` this returns an async iterator of stream events
        (`message_start` through `message_stop`), once the backend has
        accepted the request. Other keyword arguments are Messages request
        fields, such as `system`, `tools`, and `thinking`.
        """
        return await self._messages(
            self._http, model, messages, max_tokens, stream, api_base, api_key, params
        )

    def messages(
        self,
        model: str,
        messages: list[dict],
        *,
        max_tokens: int,
        stream: bool = False,
        api_base: str | None = None,
        api_key: str | None = None,
        **params: Any,
    ) -> Message | Iterator[MessageStreamEvent]:
        """The sync form of `amessages`; streams are sync iterators."""
        return self._run_sync(
            self._messages(
                self._sync_http, model, messages, max_tokens, stream, api_base, api_key, params
            ),
            stream,
        )

    async def _messages(self, http, model, messages, max_tokens, stream, api_base, api_key, params):
        body = {
            **params,
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        provider = self._provider(http, model, api_base, api_key)
        request = _translate(provider, lambda: anthropic.request_to_internal(body))
        if not stream:
            response = await provider.complete(request)
            # Messages always carries usage; a backend that reports none gets zeros.
            usage = response.usage or Usage(input_tokens=0, output_tokens=0)
            response = response.model_copy(update={"model": model, "usage": usage})
            return Message.model_validate(anthropic.response_from_internal(response))
        encoder = AnthropicStreamEncoder(model)
        events = await _open_stream(provider, request)
        return _encoded(events, encoder.encode, parse_stream_event)

    # --- plumbing -------------------------------------------------------------

    def _provider(
        self, http: HttpClients, model: str, api_base: str | None, api_key: str | None
    ) -> Provider:
        prefix, _ = split_model(model)
        if prefix not in PROVIDERS:
            raise BadRequestError(
                f"no provider for {model!r}: the library supports the prefixes "
                f"{', '.join(sorted(PROVIDERS))}"
            )
        config = copy.copy(self.config.get(prefix, {}))
        if api_base is not None:
            config["base_url"] = api_base
        if api_key is not None:
            config["api_key"] = api_key
        config["callbacks"] = [*self.callbacks, *config.get("callbacks", ())]
        try:
            return provider_for(model, {prefix: config}, http)
        except ValueError as e:
            raise BadRequestError(str(e), provider=prefix) from e

    def _run_sync(self, call, stream: bool):
        runner = self._runner()
        result = runner.run(call)
        return runner.iterate(result) if stream else result

    def _runner(self) -> LoopThread:
        with self._sync_lock:
            if self._sync is None:
                self._sync = LoopThread()
            return self._sync

    async def aclose(self) -> None:
        """Close pooled connections opened by async calls on this loop, and
        the sync loop thread if sync calls started one."""
        await self._http.aclose()
        self._close_sync()

    def close(self) -> None:
        """Close the sync loop thread and its connections. After async use,
        prefer `aclose()`, which can also close the async pool."""
        self._close_sync()

    def _close_sync(self) -> None:
        with self._sync_lock:
            runner, self._sync = self._sync, None
        if runner is not None:
            runner.run(self._sync_http.aclose())
            runner.close()

    async def __aenter__(self) -> "Client":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _translate(provider: Provider, fn: Callable[[], Request]) -> Request:
    try:
        return fn()
    except TranslationError as e:
        raise BadRequestError(str(e), provider=provider.name) from e


async def _open_stream(provider: Provider, request: Request) -> AsyncIterator[StreamEvent]:
    """Start a stream and read its first event, so a refused request raises
    before the caller starts iterating."""
    events = provider.stream(request)
    try:
        first = await anext(events)
    except StopAsyncIteration:
        raise APIResponseValidationError(
            f"{provider.name} ended the stream before it started", provider=provider.name
        ) from None
    except BaseException:
        await events.aclose()
        raise
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


async def _encoded(
    events: AsyncIterator[StreamEvent],
    encode: Callable[[StreamEvent], list[dict]],
    parse: Callable[[dict], Any],
) -> AsyncIterator[Any]:
    try:
        async for event in events:
            for item in encode(event):
                yield parse(item)
    finally:
        await events.aclose()
