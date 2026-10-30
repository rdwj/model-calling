# API Reference

Complete reference for the model-calling library and gateway.

## Client

The `Client` class provides the in-process API for calling LLM backends.

```python
from model_calling import Client

client = Client(
    config=None,
    callbacks=(),
    drop_params=False,
    models=None,
)
```

### Constructor arguments

**`config`**: dict mapping model prefix (`ollama`, `vllm`, `openai`, `openrouter`, `anthropic`, `cohere`) to settings. Each prefix accepts:

- `base_url` (str): API endpoint for this backend
- `api_key` (str): authentication key
- `headers` (dict): additional HTTP headers, sent on every call, including calls with a per-call `api_base`
- `organization` (str, `openai` only by default): sent as the `OpenAI-Organization` header, only on calls to this prefix's configured `base_url` (see [The organization header](#the-organization-header))
- `timeout` (float): request timeout in seconds, including a stalled stream (default 300)
- `callbacks` (list): callbacks that run for this prefix only, after the client-wide ones
- `transport` (httpx transport): mostly for tests

Settings merge over the defaults from `providers.default_configs()`.

**`callbacks`**: sequence of `Callback` instances. These run for every prefix, before any prefix-specific callbacks.

**`drop_params`**: bool. When True, parameters the backend cannot honor are dropped silently. When False (the default), they raise `UnsupportedParamsError`.

**`models`**: str or path-like, optional. A model alias file in the gateway's format. Names it defines are called on their targets; other names route by prefix. Reloads on change; a missing or invalid file raises `AliasConfigError` here. Needs PyYAML (the `server` extra). See [Aliases in the library](advanced.md#aliases-in-the-library).

### Config example

```python
client = Client(
    config={
        "ollama": {"base_url": "http://192.168.1.100:11434"},
        "vllm": {"base_url": "http://cluster.local:8000/v1", "api_key": "cluster-token"},
        "openai": {"api_key": "sk-..."},
    },
    drop_params=True,
)
```

### Closing

Async calls pool HTTP connections per prefix and per event loop. After async use, close the pool with `await client.aclose()` or `async with Client() as client:`.

Sync calls run on a private event-loop thread with a pool of their own. After sync use, close with `client.close()` or `with Client() as client:`.

`aclose()` closes both the async pool on the current loop and the sync thread, if sync calls started one. An unclosed client holds its connections until the process exits.

### Sync vs async

Every call method has an async form (`acompletion`, `amessages`, `aresponses`, `aembedding`) and a sync form (`completion`, `messages`, `responses`, `embedding`). Async methods return coroutines or async iterators. Sync methods return values or sync iterators, and run the call on a background thread.

One `Client` can serve both; async and sync calls keep separate pools.

### Pooling

Connections are pooled by backend prefix and reused across calls. Each prefix gets one pooled client per event loop (async) or one for the sync thread (sync). A `Client` with custom config creates its own pools; the module-level functions share a default `Client`.

## Call methods

The client provides six methods: async and sync forms of `completion` (Chat Completions), `messages` (Anthropic Messages), and `embedding` (OpenAI Embeddings).

### acompletion

```python
async def acompletion(
    model: str,
    messages: list[dict],
    *,
    stream: bool = False,
    api_base: str | None = None,
    api_key: str | None = None,
    drop_params: bool | None = None,
    timeout: float | None = None,
    extra_headers: Mapping[str, str] | None = None,
    **params,
) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]
```

Make a Chat Completions call. `model` is `prefix/name`, as `ollama/qwen3:4b`. Unprefixed names default to Ollama.

With `stream=True`, returns an async iterator of `ChatCompletionChunk` objects. The iterator returns once the backend has accepted the request, so a refusal raises here, not mid-loop.

Other keyword arguments are Chat Completions request fields, such as `temperature`, `max_tokens`, `top_p`, `stop`, `tools`, `tool_choice`, `response_format`, `stream_options`, `reasoning_effort`, and `seed`. Only one choice is supported: `n` greater than 1 is a `BadRequestError`.

Returns a `ChatCompletion` for non-streaming calls, shaped like `openai.types.chat.ChatCompletion`.

### completion

```python
def completion(
    model: str,
    messages: list[dict],
    *,
    stream: bool = False,
    api_base: str | None = None,
    api_key: str | None = None,
    drop_params: bool | None = None,
    timeout: float | None = None,
    extra_headers: Mapping[str, str] | None = None,
    **params,
) -> ChatCompletion | Iterator[ChatCompletionChunk]
```

Sync form of `acompletion`. Streams are sync iterators.

### amessages

```python
async def amessages(
    model: str,
    messages: list[dict],
    *,
    max_tokens: int,
    stream: bool = False,
    api_base: str | None = None,
    api_key: str | None = None,
    drop_params: bool | None = None,
    timeout: float | None = None,
    extra_headers: Mapping[str, str] | None = None,
    **params,
) -> Message | AsyncIterator[MessageStreamEvent]
```

Make an Anthropic Messages call over any backend. `model` is `prefix/name`.

`max_tokens` is required. With `stream=True`, returns an async iterator of stream events (`message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`). The iterator returns once the backend has accepted the request.

`messages` takes plain dicts, and the content blocks a previous call returned can go straight back as an assistant turn (`{"role": "assistant", "content": msg.content}`), as with the `anthropic` SDK. They are serialized as the SDK serializes request models.

Other keyword arguments are Messages request fields: `system` (str or list of content blocks), `tools`, `tool_choice`, `thinking` (dict with `type`, `budget_tokens`), `output_config` (its `effort` sets the reasoning effort when thinking is on, and a `format` of `{"type": "json_schema", "schema": ...}` constrains the output, sent as `response_format` or Ollama's `format`), `stop_sequences`, `metadata`, `temperature`, `top_p`, `top_k`.

On Ollama, vLLM, and OpenAI-compatible backends:

- `thinking` turns reasoning on. Ollama takes it as `think`; Chat Completions backends take `output_config.effort` as `reasoning_effort` and have no token budget.
- An effort level (`output_config.effort`, or `reasoning_effort` in `completion()`) reaches Ollama as a `think` level for models that define levels, such as gpt-oss (`"low"`, `"medium"`, `"high"`). The first such call for a model asks Ollama's `/api/show` which levels it takes, and the answer is cached for the life of the client. Other models get `think: true` and reason at their own pace.
- A `tool_result` with `is_error: true` reaches the model as its content alone. None of these backends has an error flag for tool results, so put the failure in the content.
- `metadata` and `cache_control` are ignored.

Returns a `Message` for non-streaming calls, shaped like `anthropic.types.Message`.

### messages

```python
def messages(
    model: str,
    messages: list[dict],
    *,
    max_tokens: int,
    stream: bool = False,
    api_base: str | None = None,
    api_key: str | None = None,
    drop_params: bool | None = None,
    timeout: float | None = None,
    extra_headers: Mapping[str, str] | None = None,
    **params,
) -> Message | Iterator[MessageStreamEvent]
```

Sync form of `amessages`. Streams are sync iterators.

### aresponses

```python
async def aresponses(
    model: str,
    input: str | list,
    *,
    stream: bool = False,
    api_base: str | None = None,
    api_key: str | None = None,
    drop_params: bool | None = None,
    timeout: float | None = None,
    extra_headers: Mapping[str, str] | None = None,
    **params,
) -> Response | AsyncIterator[ResponseStreamEvent]
```

Make an OpenAI Responses call over any backend. `model` is `prefix/name`. Nothing is stored: every response reports `store: false`.

`input` is a string or a list of input items: messages (text and `input_image` parts), `function_call` and `function_call_output` items, and reasoning items. The `output` of a previous call can go straight back into `input`, so a conversation carries its own history. `instructions` becomes the system prompt, ahead of any leading `system` or `developer` messages.

With `stream=True`, returns an async iterator of stream events (`response.created`, `response.in_progress`, then each output item's events, then `response.completed` or `response.incomplete`). The iterator returns once the backend has accepted the request.

Other keyword arguments are Responses request fields: `tools` (function tools), `tool_choice`, `parallel_tool_calls`, `max_output_tokens`, `temperature`, `top_p`, `reasoning` (its `effort` works as `reasoning_effort` does in `completion()`), and `text` (a `json_schema` or `json_object` format constrains the output).

The response's output items follow what the model returned: text as a `message` item, each tool call as a `function_call` item, and thinking as a `reasoning` item carrying the thinking as summary text.

These raise `BadRequestError` naming what they can't serve:

- Requests that need stored state: `previous_response_id`, `conversation`, `background`, and a stored `prompt`
- Built-in tools (`web_search`, `file_search`, `code_interpreter`, computer use) and any tool type other than `function`
- `include: ["reasoning.encrypted_content"]`, and input reasoning items carrying `encrypted_content`
- Input item and content types outside the list above, such as `input_file`

Returns a `Response` for non-streaming calls, shaped like `openai.types.responses.Response`.

### responses

```python
def responses(
    model: str,
    input: str | list,
    *,
    stream: bool = False,
    api_base: str | None = None,
    api_key: str | None = None,
    drop_params: bool | None = None,
    timeout: float | None = None,
    extra_headers: Mapping[str, str] | None = None,
    **params,
) -> Response | Iterator[ResponseStreamEvent]
```

Sync form of `aresponses`. Streams are sync iterators.

### aembedding

```python
async def aembedding(
    model: str,
    input: str | list[str] | list[int] | list[list[int]],
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    drop_params: bool | None = None,
    timeout: float | None = None,
    extra_headers: Mapping[str, str] | None = None,
    **params,
) -> CreateEmbeddingResponse
```

Make an OpenAI Embeddings call. `model` is `prefix/name`, as `ollama/nomic-embed-text`.

`input` is a string, a list of strings, a token array, or a list of token arrays. Ollama accepts text only; token arrays raise `BadRequestError`.

Other keyword arguments are Embeddings request fields: `dimensions`, `encoding_format` (`"float"` or `"base64"`), `user`.

Returns a `CreateEmbeddingResponse` shaped like `openai.types.CreateEmbeddingResponse`.

### embedding

```python
def embedding(
    model: str,
    input: str | list[str] | list[int] | list[list[int]],
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    drop_params: bool | None = None,
    timeout: float | None = None,
    extra_headers: Mapping[str, str] | None = None,
    **params,
) -> CreateEmbeddingResponse
```

Sync form of `aembedding`.

### Common parameters

All call methods accept:

- `api_base`: override the backend URL for this call
- `api_key`: override the API key for this call
- `drop_params`: override the client's `drop_params` setting for this call
- `timeout`: override the request timeout, in seconds, for this call
- `extra_headers`: HTTP headers for this call, the openai SDK's name for them, applied last over the auth header, the organization header, and the prefix's configured `headers`. Streaming calls send them too.

These take precedence over the client's config. Header names compare case-insensitively, so `extra_headers={"authorization": "Bearer other"}` replaces the configured key's `Authorization` header instead of sending two.

```python
client.completion(
    model="openai/gpt-5-mini",
    messages=[{"role": "user", "content": "hi"}],
    extra_headers={"X-Request-Id": "abc123"},
)
```

### The organization header

When `OPENAI_ORG_ID` is set, the `openai` prefix's `organization` setting takes its value, and calls to the prefix's configured `base_url` (`https://api.openai.com/v1` unless the client's config changes it) send it as `OpenAI-Organization`. A call whose `api_base` points elsewhere, such as an OpenAI-compatible vendor reached through the `openai/` prefix, leaves it off, so the organization id stays with OpenAI. An alias target's `api_base` counts as a per-call `api_base`.

To change this, set `organization` in the client's config (`""` sends none), name the header in the prefix's `headers` to send it everywhere, or pass it in `extra_headers` for one call.

## Module-level functions

The library exports eight functions at module level:

```python
from model_calling import (
    acompletion,
    amessages,
    aresponses,
    aembedding,
    completion,
    messages,
    responses,
    embedding,
)
```

These are thin wrappers over a lazily created default `Client()`. They take the same arguments as the corresponding `Client` methods. Use them for simple scripts and one-off calls. For custom config or callbacks, create your own `Client`.

To access the default client directly:

```python
from model_calling.library import default_client

client = default_client()
```

## Model prefixes and default base URLs

Model names are `prefix/name`. Unprefixed names default to `ollama`.

| Prefix | Default base_url | Credentials |
|--------|------------------|-------------|
| `anthropic` | `https://api.anthropic.com/v1` | `ANTHROPIC_API_KEY` (env) |
| `ollama` | `http://localhost:11434` | none |
| `vllm` | `http://localhost:8000/v1` | `api_key` in config if the server requires one |
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` (env), optional `OPENAI_ORG_ID` (env) |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` (env) |
| `cohere` | `https://api.cohere.com/v2` | `COHERE_API_KEY` (env) |

The library reads only these environment variables. It never reads `.env` files. The gateway (with the `server` extra) does read `.env` at startup.

Base URLs for Anthropic, vLLM, OpenAI, and OpenRouter include `/v1`, matching the openai SDK convention, and Cohere's includes `/v2`. (The `anthropic` SDK's base URL leaves it out.)

The `ollama` prefix speaks Ollama's native API (`/api/chat`, `/api/embed`), so its base URL is the server root, as `http://localhost:11434`. A base URL ending in `/v1` (with or without a trailing slash), in `config` or as a per-call `api_base`, raises `BadRequestError` before any request is sent. That path is Ollama's OpenAI-compatible API; to use it, call the model with the `openai/` prefix and `api_base="http://localhost:11434/v1"`.

The `anthropic` prefix sends Anthropic Messages to `{base_url}/messages`, so it also reaches OpenRouter's and vLLM's Messages endpoints when `base_url` points there. It authenticates with `Authorization: Bearer`, as the `anthropic` 2.x SDK does, and sends `anthropic-version: 2026-10-21`. Messages requests pass through unchanged. Chat Completions requests are translated: a missing `max_tokens` becomes 4096, `reasoning_effort` becomes adaptive thinking at that effort (OpenAI's `minimal`, which Messages lacks, sends no thinking and is reported as unsupported `thinking.effort`), `parallel_tool_calls` without tools is not sent, and `seed`, `presence_penalty`, `frequency_penalty`, and a `json_object` response format are unsupported. Thinking blocks without a signature (from another backend) are dropped from assistant turns, since Anthropic rejects them, and a turn left empty goes with them. Thinking signed by another server, such as vLLM, is sent as is and Anthropic will reject it.

The `cohere` prefix sends Cohere v2 chat requests to `{base_url}/chat` and embeddings to `{base_url}/embed`, authenticated with `Authorization: Bearer`. It has no live coverage; its tests replay fixtures written by hand from the [v2 API reference](https://docs.cohere.com/reference/chat). The system prompt becomes a `system` message, `top_p` and `top_k` become `p` and `k`, `stop` becomes `stop_sequences` (more than five raises `BadRequestError`), and thinking becomes `{type: enabled|disabled, token_budget}`. Reasoning comes back as thinking blocks, and the plan Cohere writes before its tool calls (`tool_plan`) as text. These are unsupported: a named `tool_choice`, `parallel_tool_calls: false`, `strict` on some tools but not all (Cohere's `strict_tools` covers every tool), a thinking effort without a token budget, and a response format together with tools. A finish reason of `ERROR` or `TIMEOUT` raises `APIError`. Embeddings take an `input_type` extra field (default `search_document`) and send `dimensions` as `output_dimension`, which only embed-v4 and newer accept.

## Strictness and drop_params

The library is strict by default. A parameter the backend cannot honor raises `UnsupportedParamsError` (a `BadRequestError`) before the request is sent. The exception's `.params` attribute lists the unsupported parameter names.

Examples of unsupported parameters:

- Ollama cannot force a tool call: `tool_choice` set to `"required"` or `{"type": "any"}` raises.
- Ollama cannot limit a turn to one tool call or enforce a tool's schema: `parallel_tool_calls: false` (Messages: `disable_parallel_tool_use: true`) and a tool with `strict: true` raise, as `parallel_tool_calls` and `tools.strict`.
- Chat Completions backends cannot accept `thinking.budget_tokens` (a Messages-only field) on its own. Send `output_config.effort` with it, which becomes `reasoning_effort`.
- Ollama embeddings take no fields beyond `dimensions` and `user`.

Pass `drop_params=True` (client-wide or per call) to drop unsupported parameters instead of raising.

Some parameters are always dropped silently, never an error:

- `cache_control` (a caching hint that changes no output)
- `user` for Ollama embeddings

Some requests cannot be expressed at all, regardless of `drop_params`:

- Token arrays for Ollama embeddings (`input` must be text)
- `top_k` on `openai/` models (not a Chat Completions parameter)

These raise `BadRequestError` with `status=None`.

## Response types

All responses are pydantic models shaped like the official SDK types:

- `ChatCompletion` and `ChatCompletionChunk` validate against `openai.types.chat.ChatCompletion` and `ChatCompletionChunk`
- `Message` and `MessageStreamEvent` (and its subtypes) validate against `anthropic.types.Message` and the stream events
- `Response` and `ResponseStreamEvent` (and its subtypes) validate against `openai.types.responses.Response` and `ResponseStreamEvent`
- `CreateEmbeddingResponse` validates against `openai.types.CreateEmbeddingResponse`

Access fields as attributes:

```python
response = completion(model="qwen3:4b", messages=[...])
text = response.choices[0].message.content
```

Convert to a dict with `.to_dict()`:

```python
wire = response.to_dict()
```

Unknown fields from the backend (such as vLLM's `reasoning` in `ChatCompletionMessage`) are kept and accessible as attributes. The response models are permissive: they validate the known structure and preserve everything else.

## Errors

All errors the library raises inherit from `APIError`. Catch it to handle every failure.

### Hierarchy

```
APIError
├── APIConnectionError
│   └── APITimeoutError
├── APIResponseValidationError
└── APIStatusError
    ├── BadRequestError (400, or refused before sending)
    │   └── UnsupportedParamsError
    ├── AuthenticationError (401)
    ├── PermissionDeniedError (403)
    ├── NotFoundError (404)
    ├── RateLimitError (429)
    └── InternalServerError (500+)
```

### Attributes

Every `APIError` carries:

- `message` (str): human-readable error description
- `provider` (str | None): backend prefix that raised the error (`ollama`, `vllm`, `openai`, `openrouter`)
- `status` (int | None): HTTP status code from the backend, or `None` if no usable response arrived
- `body` (Any): the backend's raw error body, if one was received

A `NotFoundError` from a backend names the request URL, as `ollama returned HTTP 404 for POST http://localhost:11434/api/chat: 404 page not found`, since a wrong base URL and a missing model can both produce a 404. URLs never carry credentials; every provider sends its key in a header. The gateway replaces the URL, like any backend address, before an error reaches its clients.

When `status` is `None`, no backend response arrived. This happens for connection failures, timeouts, requests the library refuses before sending (strict mode rejections, malformed requests), and errors a backend reports inside a stream.

### UnsupportedParamsError

A `BadRequestError` subclass raised when strict mode rejects a parameter. Adds:

- `params` (list[str]): names of the unsupported parameters

### Stream errors

Errors inside a stream (a backend reports failure mid-stream, or the connection breaks) raise from the iterator. The exception type matches the backend's error kind when it names one (rate limit, not found, invalid request). Otherwise it is `APIError` or `APIConnectionError`.

## Callbacks

Attach callbacks to observe requests, responses, and errors. Subclass `model_calling.callbacks.Callback` and override async methods.

```python
from model_calling.callbacks import Callback

class LoggingCallback(Callback):
    async def on_request_start(self, model, messages, **params):
        print(f"Calling {model}")
    
    async def on_request_end(self, model, messages, response, **params):
        print(f"Response from {model}")
```

Pass callbacks to the `Client`:

```python
client = Client(callbacks=[LoggingCallback()])
```

Or per-prefix in the config:

```python
client = Client(
    config={
        "ollama": {"callbacks": [LoggingCallback()]},
    }
)
```

Client-wide callbacks run first, then prefix-specific callbacks.

### Hooks

All hooks are async methods. A callback that raises is logged and ignored; it never changes the outcome of the request.

**`async def on_request_start(model: str, messages: list[dict], **params) -> None`**

Called before sending a request. `model` is the backend's model name (prefix stripped). `params` is the native request body minus model and messages (for Chat Completions and Messages calls).

**`async def on_request_end(model: str, messages: list[dict], response: dict, **params) -> None`**

Called after receiving a response. `response` is the backend's response body (a dict), or for a stream the last internal event (`model_calling.core.StreamEvent`).

A consumer that stops a stream early (breaking out of the loop and closing the iterator, or task cancellation) gets `on_request_end` with `cancelled=True` in `params`. This is not an error; it is the expected path for early termination.

**`async def on_stream_chunk(model: str, messages: list[dict], chunk: dict, **params) -> None`**

Called for each event in a streaming response, before it is yielded to the caller. `chunk` is the internal event (`model_calling.core.StreamEvent`), the same shape for every backend (not the raw backend line).

**`async def on_error(model: str, messages: list[dict], error: Exception, **params) -> None`**

Called when an error occurs. `error` is the exception that will be raised. A stream stopped early does not trigger this; it gets `on_request_end` with `cancelled=True`.

**`async def on_embedding_request_start(model: str, input_texts: list[str], **params) -> None`**

Called before sending an embedding request. `input_texts` is the input as a batch: a list of strings, or a list of token arrays. A single string or token array arrives as a list of one.

**`async def on_embedding_request_end(model: str, input_texts: list[str], response: dict, **params) -> None`**

Called after receiving an embedding response. `response` is the backend's response body. The `input_texts` parameter carries the same input as `on_embedding_request_start`.

### Guarantees

Every call ends with exactly one `on_request_end` or `on_error`. A stream that completes normally gets `on_request_end`. A stream stopped early gets `on_request_end(cancelled=True)`. A stream or non-stream call that fails gets `on_error`.

Callbacks run in the order they were registered (client-wide, then prefix-specific). A callback that raises does not prevent later callbacks from running.

## Gateway HTTP API

The gateway serves OpenAI Chat Completions, Anthropic Messages, and OpenAI Embeddings over HTTP. It translates requests and forwards them to backends using the library `Client`.

Start the gateway:

```bash
model-calling
```

Or:

```bash
python -m model_calling
```

The gateway listens on `0.0.0.0:8000`. Override the port with the `PORT` environment variable.

### Environment variables

- `MODEL_CALLING_CONFIG`: the model alias file (default `model-calling.yaml`); see [Model Aliases](advanced.md#model-aliases)
- `OPENAI_API_KEY`: OpenAI backend credential
- `OPENAI_ORG_ID`: OpenAI organization ID (optional)
- `OPENROUTER_API_KEY`: OpenRouter backend credential
- `ANTHROPIC_API_KEY`: Anthropic backend credential
- `PORT`: gateway listen port (default 8000)
- `MODEL_CALLING_STRICT_PARAMS`: enable strict mode (`true`, `false`, `1`, `0`, `yes`, `no`; default `false`)

The gateway reads `.env` at startup with python-dotenv.

### Endpoints

Every endpoint that takes a `model` takes an alias from the alias file. Any other name, including a backend-prefixed one such as `ollama/qwen3:4b`, gets 404 in the endpoint's error format, and responses carry the alias in `model`. The examples assume the file defines `chat` and `embed`.

#### POST /v1/chat/completions

OpenAI Chat Completions. Request and response match the OpenAI API.

Streaming: SSE (`text/event-stream`) with `data:` lines, ending with `data: [DONE]`.

Example:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chat",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

#### POST /v1/messages

Anthropic Messages. Request and response match the Anthropic API.

Streaming: SSE with `event:` and `data:` lines. Each event is named by its `type` (`message_start`, `content_block_start`, etc.).

`max_tokens` is required. Requests without it return 400.

Example:

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chat",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 512
  }'
```

#### POST /v1/messages/count_tokens

Anthropic token counting. None of the ported backends (Ollama, vLLM, OpenAI-compatible) has a token counter, so the gateway answers with a local estimate and makes no backend request. The request body is `/v1/messages` without `max_tokens` (`MessageCountTokensParams` in the `anthropic` SDK) and is validated the same way; the response is `MessageTokensCount`, `{"input_tokens": int}`.

Example:

```bash
curl -X POST http://localhost:8000/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chat",
    "messages": [{"role": "user", "content": "Hello"}],
    "system": "You are a helpful assistant.",
    "tools": [...]
  }'
```

Response:

```json
{"input_tokens": 42}
```

The estimate is about one token per four characters of text and JSON (system, messages, tool calls and results, thinking, and tool definitions), 1000 tokens per image, and a few tokens of overhead per message, block, and tool. Real counts depend on the model's tokenizer, so treat it as a rough size for budgeting context, not an exact figure.

#### POST /v1/responses

OpenAI Responses, stateless. Requests and responses match the OpenAI API within the scope `aresponses` describes, and the same requests get a 400 naming what the gateway can't serve, such as `previous_response_id` or a `web_search` tool. `store` is accepted, and every response reports `store: false`.

Streaming: SSE with `event:` and `data:` lines, each event named by its `type` and numbered by `sequence_number`. A backend failure partway through ends the stream with an `error` event.

Example:

```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chat",
    "instructions": "You are a helpful assistant.",
    "input": "Hello"
  }'
```

#### POST /v1/embeddings

OpenAI Embeddings. Request and response match the OpenAI API.

Example:

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "embed",
    "input": ["hello", "world"]
  }'
```

#### GET /v1/models

The configured aliases, in file order, as the OpenAI models list. Backends are not queried.

Response format:

```json
{
  "object": "list",
  "data": [
    {"id": "chat", "object": "model", "created": 0, "owned_by": "model-calling"},
    {"id": "embed", "object": "model", "created": 0, "owned_by": "model-calling"}
  ]
}
```

#### POST /v1/moderations

Content moderation with Granite Guardian (requires the `moderation` extra). Returns safety scores.

#### GET /

Health check. Returns `{"status": "ok"}`.

### Reserved request fields

Request bodies must not contain `api_base`, `api_key`, `drop_params`, or `extra_headers`. These are internal library parameters. The gateway returns 400 if they are present.

### Strictness

The gateway drops unsupported parameters by default. Set `MODEL_CALLING_STRICT_PARAMS=true` to turn on strict mode. In strict mode, a parameter the backend cannot honor returns 400 with message "model '`<alias>`' cannot honor `<param>`".

### Error bodies

Chat Completions and Embeddings errors use the OpenAI error format:

```json
{
  "error": {
    "message": "...",
    "type": "invalid_request_error",
    "param": null,
    "code": null
  }
}
```

Messages errors use the Anthropic error format:

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "..."
  }
}
```

Errors inside a stream are delivered as SSE events in the same format.

### Status mapping

- Backend 4xx passes through, except 401/403 (credential refused) which become 502
- Backend 5xx and unreadable responses become 502
- Timeouts become 504
- Requests rejected before sending (validation, strict mode) become 400

An error the backend reports inside a stream maps by its kind:

- Rate limit becomes 429
- Not found becomes 404
- Invalid request becomes 400
- Everything else becomes 502

### Streaming format

Chat Completions streams send `data:` lines and end with `data: [DONE]`:

```
data: {"id":"...","object":"chat.completion.chunk","choices":[...]}

data: [DONE]
```

Messages streams send `event:` and `data:` pairs. The `event` is the event type:

```
event: message_start
data: {"type":"message_start","message":{...}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{...}}

event: message_stop
data: {"type":"message_stop"}
```

### Using the official SDKs

Point the `openai` SDK at the gateway:

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="chat",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Point the `anthropic` SDK at the gateway:

```python
import anthropic

client = anthropic.Anthropic(base_url="http://localhost:8000", api_key="unused")

response = client.messages.create(
    model="chat",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=1024,
)

with client.messages.stream(
    model="chat",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=1024,
) as stream:
    for text in stream.text_stream:
        print(text, end="")
```

`messages.count_tokens` works too, with the estimate described above. The test suite drives every Messages scenario through `anthropic.AsyncAnthropic` against the gateway (`tests/test_sdk_grid.py`), including the stream helper.

The gateway does not check client API keys (pass `"unused"` or any string). It uses its own backend credentials from the environment.
