"""
The in-process library API: `Client`, with Chat Completions, Anthropic
Messages, and OpenAI Responses calls side by side, plus OpenAI Embeddings.

A client owns its per-prefix config, its callbacks, and pooled HTTP
clients. Async calls pool on the caller's event loop; sync calls run on a
private loop thread (`_sync.LoopThread`) with a pool of their own. Close a
client with `aclose()` / `async with` after async use and `close()` /
`with` after sync use.
"""

import copy
import logging
import os
import threading
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from model_calling.aliases import (
    AliasConfig,
    AliasConfigError,
    AliasStore,
    Target,
    dimension_mismatch,
)
from model_calling.callbacks import Callback
from model_calling.core import Request, Usage
from model_calling.errors import (
    APIResponseValidationError,
    BadRequestError,
    UnsupportedParamsError,
)
from model_calling.formats import TranslationError
from model_calling.formats import anthropic, embeddings, responses
from model_calling.formats import chat_completions as chat
from model_calling.formats.anthropic_stream import AnthropicStreamEncoder
from model_calling.formats.chat_completions_stream import ChatCompletionsStreamEncoder
from model_calling.formats.responses_stream import ResponsesStreamEncoder
from model_calling.library._calls import (
    complete,
    encoded,
    open_stream,
    translate,
    typed,
)
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
from model_calling.types.embeddings import CreateEmbeddingResponse
from model_calling.types.messages import Message, MessageStreamEvent, parse_stream_event
from model_calling.types.responses import Response, ResponseStreamEvent
from model_calling.types.responses import parse_stream_event as parse_response_event

logger = logging.getLogger(__name__)

PYYAML_MISSING = "Client(models=...) needs PyYAML: pip install 'model-calling[server]'"


def _warn_shadowing(config: AliasConfig) -> None:
    """Log aliases that also read as prefixed names; the alias wins."""
    shadowing = [a for a in config.aliases if "/" in a and split_model(a)[0] in PROVIDERS]
    if shadowing:
        logger.warning(
            "aliases %s shadow the prefixed names they look like; the alias is used",
            ", ".join(shadowing),
        )


def _wire(value: Any) -> Any:
    """`value` with every pydantic model in it (such as the content blocks a
    previous `messages()` call returned) dumped as a wire dict. Builds new
    containers; the caller's are untouched.

    Null fields are left out, as the wire leaves them out. Unset fields are
    kept, since a block's `type` is a default the caller never sets.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True, by_alias=True)
    if isinstance(value, (list, tuple)):
        return [_wire(v) for v in value]
    if isinstance(value, dict):
        return {k: _wire(v) for k, v in value.items()}
    return value


class Client:
    def __init__(
        self,
        *,
        config: Mapping[str, Mapping[str, Any]] | None = None,
        callbacks: Sequence[Callback] = (),
        drop_params: bool = False,
        models: str | os.PathLike | None = None,
    ):
        """`config` maps a model prefix (`anthropic`, `ollama`, `vllm`, `openai`,
        `openrouter`) to settings merged over `providers.default_configs()`:
        `base_url`, `api_key`, `headers`, `callbacks`, and `transport` (an
        httpx transport, mostly for tests). `callbacks` run for every prefix,
        before the prefix's own.

        `models` names a model alias file, the format the gateway reads (see
        `model_calling.aliases`). A name the file defines is called on its
        target, even one that looks like a prefixed name; any other name is
        routed by prefix as usual. A missing or invalid file raises
        `AliasConfigError` here; later edits reload as in the gateway.

        The call methods take `self` positionally only, so any request field,
        `self` included, can be passed as a keyword.

        A request using a parameter the backend can't honor (Ollama can't
        force a tool call) raises `BadRequestError` naming it. With
        `drop_params=True`, here or per call, it is dropped instead.
        """
        self.config = default_configs()
        for prefix, overrides in (config or {}).items():
            self.config[prefix] = {**self.config.get(prefix, {}), **overrides}
        self.callbacks = list(callbacks)
        self.drop_params = drop_params
        self._store: AliasStore | None = None
        self._aliases: AliasConfig | None = None
        if models is not None:
            try:
                import yaml  # noqa: F401 - the alias loader needs it
            except ImportError as e:
                raise ImportError(PYYAML_MISSING) from e
            path = os.fspath(models)
            if not os.path.exists(path):
                raise AliasConfigError(f"alias file {path} not found")
            self._store = AliasStore(path)
            self._aliases = self._store.check()
            _warn_shadowing(self._aliases)
        self._http = HttpClients()
        self._sync_http = HttpClients()
        self._sync: LoopThread | None = None
        self._sync_lock = threading.Lock()

    # --- Chat Completions ---------------------------------------------------

    async def acompletion(
        self,
        /,
        model: str,
        messages: list[dict],
        *,
        stream: bool = False,
        api_base: str | None = None,
        api_key: str | None = None,
        drop_params: bool | None = None,
        **params: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """A Chat Completions call. `model` is `prefix/name`, as `ollama/qwen3:4b`.

        With `stream=True` this returns an async iterator of chunks, once the
        backend has accepted the request; a refusal raises here, not mid-loop.
        Other keyword arguments are Chat Completions request fields.
        """
        return await self._completion(
            self._http, model, messages, stream, api_base, api_key, drop_params, params
        )

    def completion(
        self,
        /,
        model: str,
        messages: list[dict],
        *,
        stream: bool = False,
        api_base: str | None = None,
        api_key: str | None = None,
        drop_params: bool | None = None,
        **params: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """The sync form of `acompletion`; streams are sync iterators."""
        return self._run_sync(
            self._completion(
                self._sync_http, model, messages, stream, api_base, api_key, drop_params, params
            ),
            stream,
        )

    async def _completion(
        self, http, model, messages, stream, api_base, api_key, drop_params, params
    ):
        model, api_base, api_key, _ = self._resolve(model, api_base, api_key)
        body = {**params, "model": model, "messages": messages, "stream": stream}
        provider = self._provider(http, model, api_base, api_key)
        request = translate(provider, lambda: chat.request_to_internal(body))
        await self._check_params(provider, request, drop_params)
        if not stream:
            response = await complete(provider, request)
            return typed(
                provider,
                lambda: ChatCompletion.model_validate(chat.response_from_internal(response, model)),
            )
        encoder = ChatCompletionsStreamEncoder(model, chat.include_usage(request))
        events = await open_stream(provider, request)
        return encoded(provider, events, encoder.encode, ChatCompletionChunk.model_validate)

    # --- Anthropic Messages -------------------------------------------------

    async def amessages(
        self,
        /,
        model: str,
        messages: list[dict],
        *,
        max_tokens: int,
        stream: bool = False,
        api_base: str | None = None,
        api_key: str | None = None,
        drop_params: bool | None = None,
        **params: Any,
    ) -> Message | AsyncIterator[MessageStreamEvent]:
        """An Anthropic Messages call over any backend. `model` is `prefix/name`.

        With `stream=True` this returns an async iterator of stream events
        (`message_start` through `message_stop`), once the backend has
        accepted the request. Other keyword arguments are Messages request
        fields, such as `system`, `tools`, and `thinking`.
        """
        return await self._messages(
            self._http, model, messages, max_tokens, stream, api_base, api_key, drop_params, params
        )

    def messages(
        self,
        /,
        model: str,
        messages: list[dict],
        *,
        max_tokens: int,
        stream: bool = False,
        api_base: str | None = None,
        api_key: str | None = None,
        drop_params: bool | None = None,
        **params: Any,
    ) -> Message | Iterator[MessageStreamEvent]:
        """The sync form of `amessages`; streams are sync iterators."""
        return self._run_sync(
            self._messages(
                self._sync_http,
                model,
                messages,
                max_tokens,
                stream,
                api_base,
                api_key,
                drop_params,
                params,
            ),
            stream,
        )

    async def _messages(
        self, http, model, messages, max_tokens, stream, api_base, api_key, drop_params, params
    ):
        model, api_base, api_key, _ = self._resolve(model, api_base, api_key)
        body = {
            **params,
            "model": model,
            "messages": _wire(messages),
            "max_tokens": max_tokens,
            "stream": stream,
        }
        provider = self._provider(http, model, api_base, api_key)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            # Messages requires it; the gateway passes None when a client left it out.
            raise BadRequestError(
                f"max_tokens is required and must be a positive integer, not {max_tokens!r}",
                provider=provider.name,
            )
        request = translate(provider, lambda: anthropic.request_to_internal(body))
        await self._check_params(provider, request, drop_params)
        if not stream:
            response = await complete(provider, request)
            # Messages always carries usage; a backend that reports none gets zeros.
            usage = response.usage or Usage(input_tokens=0, output_tokens=0)
            response = response.model_copy(update={"model": model, "usage": usage})
            return typed(
                provider, lambda: Message.model_validate(anthropic.response_from_internal(response))
            )
        encoder = AnthropicStreamEncoder(model)
        events = await open_stream(provider, request)
        return encoded(provider, events, encoder.encode, parse_stream_event)

    # --- OpenAI Responses -----------------------------------------------------

    async def aresponses(
        self,
        /,
        model: str,
        input: str | list,
        *,
        stream: bool = False,
        api_base: str | None = None,
        api_key: str | None = None,
        drop_params: bool | None = None,
        **params: Any,
    ) -> Response | AsyncIterator[ResponseStreamEvent]:
        """A stateless OpenAI Responses call over any backend. `model` is
        `prefix/name`; `input` is a string or a list of input items, which
        may include a previous response's `output`.

        With `stream=True` this returns an async iterator of stream events
        (`response.created` through `response.completed`), once the backend
        has accepted the request. Other keyword arguments are Responses
        request fields, such as `instructions`, `tools`, and `reasoning`.
        Nothing is stored: a request needing stored state, such as one with
        `previous_response_id`, raises `BadRequestError` naming the field.
        """
        return await self._responses(
            self._http, model, input, stream, api_base, api_key, drop_params, params
        )

    def responses(
        self,
        /,
        model: str,
        input: str | list,
        *,
        stream: bool = False,
        api_base: str | None = None,
        api_key: str | None = None,
        drop_params: bool | None = None,
        **params: Any,
    ) -> Response | Iterator[ResponseStreamEvent]:
        """The sync form of `aresponses`; streams are sync iterators."""
        return self._run_sync(
            self._responses(
                self._sync_http, model, input, stream, api_base, api_key, drop_params, params
            ),
            stream,
        )

    async def _responses(self, http, model, input, stream, api_base, api_key, drop_params, params):
        model, api_base, api_key, _ = self._resolve(model, api_base, api_key)
        body = {**params, "model": model, "input": _wire(input), "stream": stream}
        provider = self._provider(http, model, api_base, api_key)
        request = translate(provider, lambda: responses.request_to_internal(body))
        await self._check_params(provider, request, drop_params)
        if not stream:
            response = await complete(provider, request)
            return typed(
                provider,
                lambda: Response.model_validate(
                    responses.response_from_internal(response, model, request)
                ),
            )
        encoder = ResponsesStreamEncoder(model, request)
        events = await open_stream(provider, request)
        return encoded(provider, events, encoder.encode, parse_response_event)

    # --- Embeddings -----------------------------------------------------------

    async def aembedding(
        self,
        /,
        model: str,
        input: str | list[str] | list[int] | list[list[int]],
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        drop_params: bool | None = None,
        **params: Any,
    ) -> CreateEmbeddingResponse:
        """An OpenAI Embeddings call. `model` is `prefix/name`, as
        `ollama/nomic-embed-text`. Other keyword arguments are Embeddings
        request fields: `dimensions`, `encoding_format`, `user`."""
        return await self._embedding(
            self._http, model, input, api_base, api_key, drop_params, params
        )

    def embedding(
        self,
        /,
        model: str,
        input: str | list[str] | list[int] | list[list[int]],
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        drop_params: bool | None = None,
        **params: Any,
    ) -> CreateEmbeddingResponse:
        """The sync form of `aembedding`."""
        return self._run_sync(
            self._embedding(self._sync_http, model, input, api_base, api_key, drop_params, params),
            stream=False,
        )

    async def _embedding(self, http, model, input, api_base, api_key, drop_params, params):
        model, api_base, api_key, target = self._resolve(model, api_base, api_key)
        body = {**params, "model": model, "input": input}
        provider = self._provider(http, model, api_base, api_key)
        request = translate(provider, lambda: embeddings.request_to_internal(body))
        fmt = translate(provider, lambda: embeddings.encoding_format(body))
        self._refuse(provider, provider.unsupported_embedding_params(request), drop_params)
        try:
            response = await provider.embed(request)
        except TranslationError as e:
            raise BadRequestError(str(e), provider=provider.name) from e
        payload = typed(provider, lambda: embeddings.response_from_internal(response, model, fmt))
        problem = target and dimension_mismatch(payload, target)
        if problem:
            raise APIResponseValidationError(problem, provider=provider.name)
        return typed(provider, lambda: CreateEmbeddingResponse.model_validate(payload))

    # --- plumbing -------------------------------------------------------------

    def _resolve(
        self, model: str, api_base: str | None, api_key: str | None
    ) -> tuple[str, str | None, str | None, Target | None]:
        """The prefixed model, backend, and key for `model`, through the alias
        file when it defines `model`. Explicit `api_base` and `api_key` win."""
        if self._store is None:
            return model, api_base, api_key, None
        aliases = self._store.current()
        if aliases is not self._aliases:
            self._aliases = aliases
            _warn_shadowing(aliases)
        try:
            target = aliases.resolve(model)
        except AliasConfigError as e:
            raise BadRequestError(str(e)) from e
        if target is None:
            return model, api_base, api_key, None
        return (
            target.model,
            target.api_base if api_base is None else api_base,
            target.api_key if api_key is None else api_key,
            target,
        )

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

    async def _check_params(self, provider: Provider, request: Request, drop_params: bool | None):
        """What a backend can honor may depend on its model (Ollama's
        thinking), so the provider looks the model up first."""
        await provider.prepare(request)
        self._refuse(provider, provider.unsupported_params(request), drop_params)

    def _refuse(self, provider: Provider, names: list[str], drop_params: bool | None):
        """Raise for parameters the backend can't honor, unless dropping them."""
        drop = self.drop_params if drop_params is None else drop_params
        if names and not drop:
            raise UnsupportedParamsError(
                f"{provider.name} cannot honor {', '.join(names)}; pass "
                "drop_params=True to send the request without them",
                params=names,
                provider=provider.name,
            )

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
            runner.close(final=self._sync_http.aclose)

    async def __aenter__(self) -> "Client":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
