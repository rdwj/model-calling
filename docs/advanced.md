# Advanced Topics

This page covers in-process library usage: observability, connection pooling, testing, concurrency, and running the gateway in production.

## Observability with Callbacks

The `model_calling.callbacks.Callback` interface lets you add logging, tracing, or metrics without changing your calling code. Subclass it, override the async hooks you need, and pass instances to the client.

### Basic Logging Callback

```python
import logging
from model_calling import Client
from model_calling.callbacks import Callback

logger = logging.getLogger(__name__)


class LoggingCallback(Callback):
    async def on_request_start(self, model, messages, **params):
        logger.info(f"Calling {model} with {len(messages)} messages")

    async def on_request_end(self, model, messages, response, **params):
        cancelled = params.get("cancelled", False)
        if cancelled:
            logger.warning(f"{model} stream stopped early")
        else:
            logger.info(f"{model} completed")

    async def on_error(self, model, messages, error, **params):
        logger.error(f"{model} failed: {error}")


client = Client(callbacks=[LoggingCallback()])
```

### Timing and Metrics Callback

A callback that tracks latency. Start and end of one call receive the same
`messages` (or `input`) object, so its `id()` pairs them:

```python
import time
from model_calling.callbacks import Callback


class MetricsCallback(Callback):
    def __init__(self):
        self.call_times = {}

    async def on_request_start(self, model, messages, **params):
        self.call_times[id(messages)] = time.monotonic()

    async def on_request_end(self, model, messages, response, **params):
        # response is the backend's own body (a dict in its native format) for
        # a non-streaming call, or the last internal event for a stream.
        elapsed = time.monotonic() - self.call_times.pop(id(messages), time.monotonic())
        print(f"{model}: {elapsed:.2f}s")

    async def on_embedding_request_start(self, model, input, **params):
        call_id = id(input)
        self.call_times[call_id] = time.monotonic()

    async def on_embedding_request_end(self, model, input, response, **params):
        call_id = id(input)
        elapsed = time.monotonic() - self.call_times.pop(call_id, time.monotonic())
        count = len(input) if isinstance(input, list) else 1
        print(f"{model} embeddings: {elapsed:.2f}s for {count} inputs")
```

### Streaming and Cancellation

A callback that handles streams and early stops:

```python
from model_calling.callbacks import Callback


class StreamingCallback(Callback):
    async def on_stream_chunk(self, model, messages, chunk, **params):
        # chunk is a model_calling.core.StreamEvent, the same shape
        # for every backend
        print(f"Event: {chunk.type}")

    async def on_request_end(self, model, messages, response, **params):
        if params.get("cancelled"):
            print(f"{model} stream stopped early by the consumer")
```

When a consumer breaks out of a stream or the task is cancelled, `on_request_end` gets `cancelled=True` instead of `on_error`. Every call ends with exactly one `on_request_end` or `on_error`.

### Per-Prefix Callbacks

Attach callbacks to specific prefixes through config:

```python
from model_calling import Client

client = Client(
    config={
        "ollama": {
            "callbacks": [LoggingCallback()],
        },
        "openai": {
            "callbacks": [MetricsCallback()],
        },
    },
    callbacks=[StreamingCallback()],  # runs for every prefix
)
```

Client-wide callbacks run first, then the prefix's own.

### Error Handling

A callback that raises is logged and ignored. Callbacks never change the request's outcome: a callback error won't turn a successful call into a failure, and a callback can't suppress a real error.

## Connection Pooling and Client Lifecycle

One `Client` per application. The client pools connections per prefix and per event loop, reusing them across calls.

### Async Use

Close the client after async use to release its connections:

```python
from model_calling import Client


async def main():
    client = Client()
    try:
        result = await client.acompletion(
            model="ollama/qwen3:4b",
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(result.choices[0].message.content)
    finally:
        await client.aclose()
```

Or use it as a context manager:

```python
async def main():
    async with Client() as client:
        result = await client.acompletion(
            model="ollama/qwen3:4b",
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(result.choices[0].message.content)
```

### Sync Use

The sync API runs on a private event-loop thread with its own connection pool. Close the client to shut down the thread and release its connections:

```python
from model_calling import Client


def main():
    client = Client()
    try:
        result = client.completion(
            model="ollama/qwen3:4b",
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(result.choices[0].message.content)
    finally:
        client.close()
```

Or as a context manager:

```python
def main():
    with Client() as client:
        result = client.completion(
            model="ollama/qwen3:4b",
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(result.choices[0].message.content)
```

### Multiple Event Loops

An async client's connections belong to the event loop that opened them. Each loop gets its own pool. A script calling `asyncio.run` twice, or threads each running a loop, never share connections:

```python
import asyncio
from model_calling import Client


async def first_loop():
    async with Client() as client:
        result = await client.acompletion(
            model="ollama/qwen3:4b",
            messages=[{"role": "user", "content": "First loop"}],
        )
        print(result.choices[0].message.content)


async def second_loop():
    async with Client() as client:
        # This gets its own connection pool
        result = await client.acompletion(
            model="ollama/qwen3:4b",
            messages=[{"role": "user", "content": "Second loop"}],
        )
        print(result.choices[0].message.content)


asyncio.run(first_loop())
asyncio.run(second_loop())
```

A finished loop's connections go with it unclosed, since their loop can no longer close them.

### Using Sync API from Threads

The sync API is safe to call from multiple threads. Each client's sync calls share one background loop thread and one connection pool:

```python
import threading
from model_calling import Client


def worker(client, name):
    result = client.completion(
        model="ollama/qwen3:4b",
        messages=[{"role": "user", "content": f"Hello from {name}"}],
    )
    print(f"{name}: {result.choices[0].message.content}")


client = Client()
try:
    threads = [threading.Thread(target=worker, args=(client, f"thread-{i}")) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
finally:
    client.close()
```

## Per-Call Overrides

Override the base URL or API key for a single call:

```python
from model_calling import Client


async def main():
    client = Client()
    
    # Use a different vLLM server for this call
    result = await client.acompletion(
        model="vllm/gpt-oss-20b",
        messages=[{"role": "user", "content": "Hello"}],
        api_base="http://other-host:8000/v1",
    )
    
    # Use a different OpenAI key
    result = await client.acompletion(
        model="openai/gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        api_key="sk-different-key",
    )
```

### Custom Headers

Add custom headers through config:

```python
from model_calling import Client

client = Client(
    config={
        "openai": {
            "headers": {
                "X-Request-ID": "abc123",
                "X-Custom-Header": "value",
            },
        },
    },
)
```

Headers go on every request to that prefix, after auth.

## Testing Offline

Pass an `httpx.MockTransport` through config to test without network calls. The
mock answers in the backend's native format; here the `vllm` prefix, which speaks
OpenAI Chat Completions:

```python
import httpx
from model_calling import Client


def handler(request: httpx.Request) -> httpx.Response:
    """Mock backend that returns a fixed response"""
    body = {
        "id": "test-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-oss-20b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Mocked response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return httpx.Response(200, json=body)


async def test_completion():
    config = {"vllm": {"transport": httpx.MockTransport(handler)}}
    async with Client(config=config) as client:
        result = await client.acompletion(
            model="vllm/gpt-oss-20b",
            messages=[{"role": "user", "content": "Hello"}],
        )
    assert result.choices[0].message.content == "Mocked response"
```

The transport is fixed when the client is built, so changing it later needs a new `Client`.

## Concurrency

Process multiple requests concurrently with `asyncio.gather`:

```python
import asyncio
from model_calling import Client


async def main():
    client = Client()
    
    tasks = [
        client.acompletion(
            model="ollama/qwen3:4b",
            messages=[{"role": "user", "content": f"Request {i}"}],
        )
        for i in range(10)
    ]
    
    results = await asyncio.gather(*tasks)
    for i, result in enumerate(results):
        print(f"Request {i}: {result.choices[0].message.content}")


asyncio.run(main())
```

### Rate Limiting with a Semaphore

Limit concurrent requests to avoid overwhelming a backend:

```python
import asyncio
from model_calling import Client


async def call_with_limit(client, semaphore, query):
    async with semaphore:
        return await client.acompletion(
            model="ollama/qwen3:4b",
            messages=[{"role": "user", "content": query}],
        )


async def main():
    client = Client()
    semaphore = asyncio.Semaphore(5)  # max 5 concurrent requests
    
    tasks = [
        call_with_limit(client, semaphore, f"Request {i}")
        for i in range(20)
    ]
    
    results = await asyncio.gather(*tasks)
    print(f"Completed {len(results)} requests")


asyncio.run(main())
```

## Running the Gateway in Production

The gateway is a FastAPI application that serves the library over HTTP. It needs the `[server]` extra:

```bash
pip install 'model-calling[server]'
```

### Starting the Gateway

Start it with the `model-calling` command:

```bash
model-calling
```

Or run the module directly:

```bash
python -m model_calling
```

The gateway binds to `0.0.0.0:8000` by default. Set the `PORT` environment variable to change it:

```bash
PORT=8080 model-calling
```

With `uvicorn` directly:

```bash
uvicorn model_calling.server:app --host 0.0.0.0 --port 8000
```

### Model Aliases

Gateway clients name models by alias, never by backend. A YAML file maps each alias to a backend and that backend's model name, so moving an app from Ollama in development to vLLM in production is an edit to this file, with no change to the app:

```yaml
backends:
  prod:
    provider: vllm
    base_url: https://vllm.apps.cluster.example.com/v1
    api_key_env: VLLM_API_KEY   # the variable holding the key, never the key
models:
  chat: {backend: prod, model: gpt-oss-20b}
  # A provider name works as a backend, using the defaults below.
  embed: {backend: ollama, model: nomic-embed-text, dimensions: 768}
```

`backends` is optional. A backend has a `provider` (`ollama`, `vllm`, `openai`, `openrouter`, or `anthropic`), and optionally a `base_url` and an `api_key_env`. A model has a `backend`, a `model`, and for embeddings an optional `dimensions`. With `dimensions` set, the gateway answers 502 rather than pass on vectors of another size, so a backend swap can't silently corrupt a vector index. Alias names are free-form: an app written against vLLM can keep the name vLLM serves, such as `openai/gpt-oss-20b`.

The gateway reads the file named by `MODEL_CALLING_CONFIG`, or `model-calling.yaml` in its working directory. It checks the file's modification time, size, and inode on every request and reloads it when they change; no restart is needed. Replace the file atomically (write a new file and rename it over the old one, as editors and OpenShift ConfigMap mounts do): an in-place rewrite can be read half-written, and two in-place edits of the same size within the filesystem's timestamp resolution look like one. An edit that fails to load leaves the previous models serving and logs the error once. A file that is invalid at startup stops the gateway. Without a file, no models are served.

Clients see only aliases. `GET /v1/models` lists them, responses carry the alias in `model`, and any other name gets 404; a `prefix/model` string such as `ollama/qwen3:4b` is only served if the file defines it as an alias. Error messages name the alias, never the backend, its model name, or its address. Cohere is not available yet.

#### Aliases in the library

`Client` reads the same file when you pass its path, so an app that calls the library directly can use the names its gateway clients use:

```python
from model_calling import Client

client = Client(models="model-calling.yaml")
response = await client.acompletion("chat", messages)
```

A name the file defines is called on its target; any other name is routed by its prefix as usual, so `ollama/qwen3:4b` still works. If the file defines an alias that is itself prefix-shaped, such as `ollama/qwen3:4b`, the alias wins, and the client logs a warning naming it when the file loads. An explicit `api_base` or `api_key` on a call overrides the alias's backend. Unlike the gateway, the library returns the backend's model name in `model`, not the alias.

The file reloads on change as it does in the gateway. A missing or invalid file raises `AliasConfigError` when the client is created, a backend whose `api_key_env` variable is unset raises `BadRequestError` on the call, and an embedding of the wrong size for an alias with `dimensions` raises `APIResponseValidationError`. Reading the file needs PyYAML, which the `server` extra installs.

### Environment Variables

The `model-calling` command (and `python -m model_calling`) reads a `.env` file from the directory it starts in, via python-dotenv, before the gateway builds its backend settings. Importing `model_calling.server`, as `uvicorn model_calling.server:app` does, never reads `.env`, and neither does the library.

- `OPENAI_API_KEY`, `OPENAI_ORG_ID`: OpenAI credentials
- `OPENROUTER_API_KEY`: OpenRouter key
- `ANTHROPIC_API_KEY`: Anthropic key
- `MODEL_CALLING_CONFIG`: the alias file (default `model-calling.yaml`)
- `OLLAMA_BASE_URL`: Ollama's native API root, without `/v1` (default `http://localhost:11434`)
- `VLLM_BASE_URL`: vLLM's API root with `/v1` (default `http://localhost:8000/v1`)
- `VLLM_API_KEY`: sent to vLLM as a bearer token, if set
- `MODEL_CALLING_ANTHROPIC_BASE_URL`: the Anthropic backend's API root with `/v1` (default `https://api.anthropic.com/v1`). `ANTHROPIC_API_KEY`, if set, goes to this host.
- `PORT`: server port (default 8000)
- `MODEL_CALLING_STRICT_PARAMS`: strict mode (see below)

The URL variables mean what `base_url` means in `Client(config=...)`. They are the defaults for a backend the alias file gives no `base_url`; the file wins where both set one. The gateway logs the models it serves and these URLs at startup. The Anthropic variable is not `ANTHROPIC_BASE_URL` because the `anthropic` SDK and Claude Code read that name as the root without `/v1`, and a client pointed at this gateway sets it to the gateway's own address.

### Strict Mode

By default the gateway drops parameters a backend can't honor. Set `MODEL_CALLING_STRICT_PARAMS=true` to turn on strict mode: the gateway returns 400 with a message naming the unsupported parameter instead.

```bash
MODEL_CALLING_STRICT_PARAMS=true model-calling
```

Accepts `true`, `false`, `1`, `0`, `yes`, `no`. Anything else fails at startup.

### Reserved Fields

Request bodies must not contain `api_base`, `api_key`, or `drop_params`. The gateway refuses them with 400. These are library-only keyword arguments that would send the gateway's credentials elsewhere or override its parameter policy.

### Status Mapping

Backend errors map to HTTP status codes the gateway client sees:

- Backend 4xx passes through, except 401 and 403 (the gateway's credential was refused) which become 502
- Backend 5xx and unreadable responses become 502
- Timeouts become 504
- Requests the library refuses before sending become 400

An error the backend reports inside a stream maps by its kind: rate limit becomes 429, not found becomes 404, invalid request becomes 400, anything else becomes 502.

### Building Containers

The gateway ships with `Containerfile` and `Containerfile-FIPS` (UBI9 base). Build with podman:

```bash
podman build -t model-calling:latest -f Containerfile .
```

For OpenShift deployment from a Mac, specify the platform:

```bash
podman build --platform linux/amd64 -t model-calling:latest -f Containerfile .
```

The FIPS variant runs `fips-mode-setup`, sets the FIPS crypto policy, and sets FIPS-related environment variables. It has not been validated yet; see `fips-todo.md`.

```bash
podman build --platform linux/amd64 -t model-calling:fips -f Containerfile-FIPS .
```

### Podman Compose

The `podman-compose.yml` file runs the gateway alongside Ollama:

```bash
podman-compose up
```

The file sets `OLLAMA_BASE_URL=http://ollama:11434`, so the gateway reaches the Ollama container with no further setup. The file uses digest-pinned images and bind mounts Ollama's model storage.

## Not Provided

The library does not provide retries, fallbacks, load balancing, or cost tracking. Callers can wrap `Client` calls themselves. A retry example:

```python
import asyncio
from model_calling import Client
from model_calling.errors import RateLimitError


async def call_with_retry(client, model, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await client.acompletion(model, messages)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"Rate limited, retrying in {wait}s")
            await asyncio.sleep(wait)
```

Load balancing across multiple gateway instances is the caller's responsibility. Standard HTTP load balancers work.
