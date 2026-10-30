# Migrating from LiteLLM

model-calling's library API mirrors a few LiteLLM call patterns on purpose: module-level `completion()` and `acompletion()` functions, `"provider/model"` model strings, and a `drop_params` switch. This guide lists the places where model-calling differs by design, so a LiteLLM user knows what to change and what to test.

The guide also covers upgrading from model-calling 0.1.x, at the end, and moving an application from local development to production.

## Contents

- [Coming from LiteLLM](#coming-from-litellm)
- [From Local Development to Production](#from-local-development-to-production)
- [Checklist](#checklist)
- [Upgrading from 0.1.x](#upgrading-from-01x)

## Coming from LiteLLM

### What carries over

A call in that shape:

```python
import model_calling

response = model_calling.completion(
    model="openrouter/mistralai/mistral-nemo",
    messages=[{"role": "user", "content": "Hello"}],
    drop_params=True,
)
print(response.choices[0].message.content)
```

`completion`, `messages`, `responses`, and `embedding`, with their async forms `acompletion`, `amessages`, `aresponses`, and `aembedding`, exist at module level and run on a default `Client` created on first use. For your own configuration, callbacks, or control over when connections close, create a `Client` and call the same methods on it.

### Differences at a glance

| Topic | model-calling |
|-------|---------------|
| Unsupported parameters | The library raises `UnsupportedParamsError` unless you pass `drop_params=True`. The gateway drops them unless strict mode is on. |
| Model strings | A fixed set of prefixes: `ollama/`, `vllm/`, `openai/`, `openrouter/`, `anthropic/`, `cohere/`, `openrouter_responses/`. Any other prefix raises `BadRequestError`. A name with no prefix goes to Ollama. |
| Responses | Typed pydantic objects shaped like the official `openai` and `anthropic` SDK types. Use attributes, or `to_dict()` for a plain dict. |
| Errors | One hierarchy rooted at `model_calling.APIError`, named after the `openai` SDK's exceptions. |
| Timeout | 300 seconds by default, a stalled stream included. Set it per call or per prefix. |
| API formats | `messages()` (Anthropic Messages) and `responses()` (OpenAI Responses, stateless) sit alongside `completion()`, over every backend. |
| Configuration | The library reads only backend API keys and the OpenAI organization ID from the environment, and never reads `.env`. |
| Gateway | Clients name models by alias, never by backend. Bodies may not carry `api_base`, `api_key`, or `drop_params`. JSON keys follow the SDK types' order. |
| Not provided | Retries, fallbacks, load balancing, and cost tracking. |

The sections below take these one at a time.

### Strict by default

The library refuses a parameter the backend cannot honor. Before the request is sent, the call raises `UnsupportedParamsError`, a subclass of `BadRequestError`, and its `params` attribute names what was refused. Ollama, for example, cannot force a tool call:

```python
import model_calling

weather_tool = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
}

try:
    model_calling.completion(
        "ollama/qwen3:4b",
        [{"role": "user", "content": "What's the weather?"}],
        tools=[weather_tool],
        tool_choice="required",
    )
except model_calling.UnsupportedParamsError as e:
    print(e.params)  # ['tool_choice']
```

Pass `drop_params=True` to a single call, or to `Client(drop_params=True)` for every call, and the request goes out without the parameter. A caching hint (`cache_control`) changes no output, so it is always dropped silently, even in strict mode.

Some requests are refused whatever `drop_params` says, because the library or that backend cannot express them: `n` greater than 1, the deprecated `functions` and `function_call` fields (send `tools`), and `top_k` on `openai/` models. These raise `BadRequestError` with `status=None`.

The gateway takes the opposite default. It drops unsupported parameters unless its environment sets `MODEL_CALLING_STRICT_PARAMS=true`, in which case the client gets a 400 naming the parameter.

### Model prefixes

The prefix picks the backend, and the rest of the string is the backend's own model name:

| Prefix | Backend | Example |
|--------|---------|---------|
| `ollama/` | Ollama's native API | `ollama/qwen3:4b` |
| `vllm/` | vLLM's OpenAI-compatible API | `vllm/gpt-oss-20b` |
| `openai/` | OpenAI | `openai/gpt-5` |
| `openrouter/` | OpenRouter Chat Completions | `openrouter/mistralai/mistral-nemo` |
| `anthropic/` | Anthropic Messages, at Anthropic or any server that serves `{base_url}/messages` | `anthropic/claude-haiku-4-5` |
| `cohere/` | Cohere v2 | `cohere/command-a` |
| `openrouter_responses/` | OpenRouter's Responses API | `openrouter_responses/mistralai/mistral-nemo` |

The string splits at the first `/`, so OpenRouter names keep their own slash. A name with no prefix goes to Ollama. Any other prefix raises `BadRequestError` naming the supported ones, before a request is sent. There is no catch-all prefix for other OpenAI-compatible servers; use `openai/` or `vllm/` with `api_base` pointed at the server. OpenAI-compatible base URLs include `/v1`, as the `openai` SDK's do.

A response's `model` field carries the prefixed backend name, such as `ollama/qwen3:4b`.

The library can also read the gateway's alias file: with `Client(models="model-calling.yaml")`, a name the file defines is called on its target, and any other name is routed by prefix (see [Model Aliases](advanced.md#model-aliases)). A response to an alias call carries the backend name in `model`, not the alias.

### Typed responses

Every call returns a pydantic model shaped like the official SDK type for its format: `ChatCompletion` and `ChatCompletionChunk` for `completion()`, `Message` and its stream events for `messages()`, `Response` and its stream events for `responses()`, and `CreateEmbeddingResponse` for `embedding()`. The test suite validates this output against the `openai` and `anthropic` SDK types.

Read fields as attributes. Code that indexes a response as a dict (`response["choices"]`) needs attributes instead, or a call to `to_dict()` first:

```python
from model_calling import Client

with Client() as client:
    response = client.completion("ollama/qwen3:4b", [{"role": "user", "content": "Hi"}])
    print(response.choices[0].message.content)
    wire = response.to_dict()  # a plain dict, shaped as on the wire
```

Fields a backend adds beyond the SDK type, such as vLLM's `reasoning`, are kept and readable as attributes. Content blocks from a `messages()` response can go straight back into the next call's `messages`, with no conversion.

### Errors

Every exception the library raises subclasses `model_calling.APIError`. The subclasses take the `openai` SDK's names:

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

Each carries `message`, `provider` (the prefix that raised it), `status` (the backend's HTTP status), and `body` (the backend's raw error body). `status` is `None` when no backend response arrived: a connection failure, a timeout, a request the library refused before sending, or an error the backend reported inside a stream. A stream error that names its kind (rate limit, not found, invalid request) raises the matching class.

A backend that sends a body the library cannot express as the SDK type raises `APIResponseValidationError`, so malformed output fails at the call.

### The request timeout

Requests time out after 300 seconds by default. Pass `timeout` (seconds) to a call, or set it per prefix in the client's config; the per-call value wins:

```python
from model_calling import Client

client = Client(config={"vllm": {"timeout": 60.0}})
response = client.completion(
    "vllm/gpt-oss-20b",
    [{"role": "user", "content": "Hello"}],
    timeout=10.0,
)
client.close()
```

A request that times out raises `APITimeoutError`. Once a stream is under way, a stall raises `APIConnectionError` from the iterator, since the connection is treated as broken.

### messages() and responses() alongside completion()

`completion()` speaks OpenAI Chat Completions. Two more call families speak other formats over the same backends, with the same sync and async forms and the same prefixes:

- `messages()` and `amessages()` take and return Anthropic Messages. `max_tokens` is a required keyword argument, as in the Messages API, and a call without a positive integer raises `BadRequestError`.
- `responses()` and `aresponses()` take and return OpenAI Responses, without a response store. A request that needs stored state, such as one with `previous_response_id`, or a built-in tool such as `web_search`, raises `BadRequestError` naming the field. To continue a conversation, send the previous response's `output` back in `input`.

```python
from model_calling import Client

with Client() as client:
    message = client.messages(
        "ollama/qwen3:4b",
        [{"role": "user", "content": "Hello"}],
        max_tokens=256,
    )
    print(next(b.text for b in message.content if b.type == "text"))

    reply = client.responses("ollama/qwen3:4b", "Hello", instructions="Be brief.")
    print(reply.output_text)
```

The library translates between formats as needed, so `messages()` works against Ollama or OpenAI and `completion()` works against an `anthropic/` backend. A `completion()` call to `anthropic/` without `max_tokens` sends 4096, since Messages requires a value.

### Streaming calls

With `stream=True`, `acompletion()` (and the other async methods) must still be awaited. The await returns an async iterator once the backend has accepted the request, so a refused request raises at the call, not inside the loop. The sync methods return a plain iterator.

### Configuration and environment

The library reads backend API keys (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `COHERE_API_KEY`) and the OpenAI organization ID (`OPENAI_ORG_ID`) from the environment, and nothing else. Backend URLs come from built-in defaults, the `config` dict passed to `Client`, or per-call `api_base`. Importing the library never reads a `.env` file; the gateway's `model-calling` command does, at startup.

```python
from model_calling import Client

client = Client(
    config={
        "vllm": {"base_url": "https://vllm.example.com/v1", "api_key": "cluster-token"},
    }
)
```

Callbacks are `model_calling.callbacks.Callback` subclasses with async hooks (`on_request_start`, `on_request_end`, `on_stream_chunk`, `on_error`, and two embedding hooks). Pass them to `Client(callbacks=[...])` for every prefix, or under a prefix in `config` for that prefix only. See [Callbacks](api-reference.md#callbacks).

A `Client` pools connections. Close it with `close()` or `with` after sync use, and `aclose()` or `async with` after async use.

### The gateway

The gateway serves Chat Completions, Messages, Responses, and Embeddings over HTTP for the official `openai` and `anthropic` SDKs. Several of its rules differ from what a library user might expect.

**Clients name models by alias.** A YAML file (`MODEL_CALLING_CONFIG`, default `model-calling.yaml`) maps each alias to a backend and model, and the gateway serves only those aliases. Any other name gets 404, a `prefix/model` name included. Responses carry the alias in `model`, and error messages replace the backend's name, model, and URLs, so no backend name crosses the wire in either direction. See [Model Aliases](advanced.md#model-aliases).

```yaml
models:
  chat: {backend: ollama, model: qwen3:4b}
```

**Some body fields are refused.** A request body containing `api_base`, `api_key`, or `drop_params` gets a 400 naming the fields. Honoring them would let a client send the gateway's credentials to another host or override its parameter policy.

**Unsupported parameters are dropped by default.** Set `MODEL_CALLING_STRICT_PARAMS=true` for strict mode (see [above](#strict-by-default)).

**Client keys are not checked.** The gateway calls backends with keys from its own environment, which the alias file can name per backend (`api_key_env`). SDK clients can pass any placeholder key.

**JSON keys follow the SDK types.** Non-streaming responses list their keys in the order the `openai` and `anthropic` SDK types declare them. Clients that parse JSON are unaffected; anything that compares raw response text may need updating.

**Statuses are the gateway's.** A backend 4xx passes through, except that a backend's 401 or 403 (it refused the gateway's credential) becomes 502. Backend 5xx statuses and unreadable responses become 502, and a timeout becomes 504. `POST /v1/messages/count_tokens` answers with a local estimate, since no backend is asked.

### Not provided

The library does not retry, fall back to another model, balance load, or track cost. Each call makes one request to one backend and raises on failure. Wrap calls yourself where you need these; [Not Provided](advanced.md#not-provided) has a retry example.

## From Local Development to Production

A typical workflow starts with Ollama for local iteration, moves to a hosted service (OpenAI, OpenRouter) or vLLM on OpenShift for shared testing, and deploys to production with environment-specific model names and authentication.

The library reads from the environment only the variables listed under [Configuration and environment](#configuration-and-environment). It never reads a `.env` file. The gateway loads `.env` with python-dotenv at startup. The library's per-prefix `config` takes `base_url`, `api_key`, `headers`, and `timeout`. The gateway configures a backend through its alias file (`provider`, `base_url`, and `api_key_env`) and a few environment variables, and has no per-backend headers setting.

### Scenario: Ollama to vLLM on OpenShift

**Local development:**

```python
from model_calling import Client

client = Client()
response = client.completion(
    model="ollama/qwen3:4b",
    messages=[{"role": "user", "content": "Summarize this document."}],
    temperature=0.7,
)
print(response.choices[0].message.content)
client.close()
```

**Production (vLLM on OpenShift):**

```python
import os
from model_calling import Client

# Read your app's own config (the library reads only its own few env vars)
vllm_url = os.environ["VLLM_BASE_URL"]  # e.g. "https://vllm.apps.cluster.example.com/v1"
vllm_token = os.environ["VLLM_API_KEY"]

client = Client(
    config={
        "vllm": {
            "base_url": vllm_url,
            "api_key": vllm_token,
        }
    }
)

response = client.completion(
    model="vllm/gpt-oss-20b",  # The model the vLLM server serves
    messages=[{"role": "user", "content": "Summarize this document."}],
    temperature=0.7,
)
print(response.choices[0].message.content)
client.close()
```

The message format, tools, and response processing stay the same. Only the model name and configuration change.

### Scenario: Ollama to OpenRouter

Local calls to `ollama/mistral-nemo` become `openrouter/mistralai/mistral-nemo` in production. Set `OPENROUTER_API_KEY` in the environment or pass it through `config`.

```python
from model_calling import Client

# OPENROUTER_API_KEY from environment
client = Client()

response = client.completion(
    model="openrouter/mistralai/mistral-nemo",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Per-call overrides work for one-off calls that need a different key or endpoint:

```python
response = client.completion(
    model="openrouter/mistralai/mistral-nemo",
    messages=[{"role": "user", "content": "Hello"}],
    api_key="sk-or-v1-...",  # Override for this call
)
```

### Environment-based configuration

Applications that run in dev, test, and prod can read their own environment variables and build the client config from them. The library itself reads only the environment variables listed under [Configuration and environment](#configuration-and-environment).

```python
import os
from model_calling import Client

env = os.environ.get("DEPLOY_ENV", "dev")

if env == "dev":
    model = "ollama/qwen3:4b"
    config = {}  # Defaults to localhost
elif env == "test":
    model = "vllm/gpt-oss-20b"
    config = {
        "vllm": {
            "base_url": os.environ["VLLM_TEST_URL"],
            "api_key": os.environ["VLLM_TEST_TOKEN"],
        }
    }
else:  # production
    model = "vllm/gpt-oss-20b"
    config = {
        "vllm": {
            "base_url": os.environ["VLLM_PROD_URL"],
            "api_key": os.environ["VLLM_PROD_TOKEN"],
        }
    }

client = Client(config=config)
response = client.completion(model=model, messages=[...])
client.close()
```

### Using the gateway

A gateway serves many applications. Each app points the `openai` or `anthropic` SDK at the gateway and names models by alias, and the gateway holds the backend configurations and keys. Its alias file maps each alias to a backend (see [Model Aliases](advanced.md#model-aliases)); for backends the file gives no URL, it reads URLs (`VLLM_BASE_URL`, `OLLAMA_BASE_URL`) and keys (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `VLLM_API_KEY`) from its environment. In OpenShift, mount the keys from Secrets:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: model-calling-keys
type: Opaque
data:
  OPENROUTER_API_KEY: <base64-encoded-key>
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-calling-gateway
spec:
  template:
    spec:
      containers:
      - name: gateway
        image: quay.io/your-org/model-calling:latest
        env:
        - name: VLLM_BASE_URL
          value: https://vllm.apps.cluster.example.com/v1
        - name: OPENROUTER_API_KEY
          valueFrom:
            secretKeyRef:
              name: model-calling-keys
              key: OPENROUTER_API_KEY
```

Applications call the gateway with no authentication (the gateway uses its own backend credentials):

```python
import openai

client = openai.OpenAI(
    base_url="http://model-calling-gateway:8000/v1",
    api_key="unused",
)
response = client.chat.completions.create(
    model="chat",  # served from whichever backend the alias file names
    messages=[{"role": "user", "content": "Hello"}],
)
```

The same gateway serves Chat Completions (`POST /v1/chat/completions`), Anthropic Messages (`POST /v1/messages`), OpenAI Responses without a response store (`POST /v1/responses`), and Embeddings (`POST /v1/embeddings`).

### Anthropic Messages over any backend

The `messages` method translates Anthropic Messages requests to the backend's format. This works over Ollama, vLLM, OpenAI, OpenRouter, Anthropic, and (from 0.3.0) Cohere.

```python
from model_calling import Client

client = Client()

# Anthropic Messages over Ollama
response = client.messages(
    model="ollama/qwen3:4b",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
# Response may include thinking blocks; extract the text block
text = next(b.text for b in response.content if b.type == "text")
print(text)

# Same call, different backend
response = client.messages(
    model="openrouter/mistralai/mistral-nemo",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
text = next(b.text for b in response.content if b.type == "text")
print(text)
```

The gateway's `/v1/messages` endpoint accepts Anthropic Messages requests and returns Anthropic-shaped responses and stream events, translating them to and from the backend.

### Streaming

Async streams are async iterators. Sync streams are plain iterators. A streaming call returns only once the backend has accepted the request, so a refused request raises at the call, not mid-loop.

```python
import asyncio
from model_calling import Client

async def stream_example():
    client = Client()
    
    stream = await client.acompletion(
        model="ollama/qwen3:4b",
        messages=[{"role": "user", "content": "Count to 10"}],
        stream=True,
    )
    
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()
    
    await client.aclose()

asyncio.run(stream_example())
```

Close async clients with `await client.aclose()` or `async with Client() as client`. Close sync clients with `client.close()` or `with Client() as client`.

### Complete example: tool use across environments

This agent calls a weather function. The code is the same in dev (Ollama) and production (vLLM on OpenShift); only the model name and configuration change.

```python
import os
import json
from model_calling import Client

def get_weather(location: str) -> dict:
    return {"location": location, "temperature": 22, "unit": "celsius"}

weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"],
        },
    },
}

def run_agent(model: str, config: dict):
    client = Client(config=config)
    
    response = client.completion(
        model=model,
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[weather_tool],
    )
    
    message = response.choices[0].message
    if not message.tool_calls:
        print(message.content)
        client.close()
        return
    
    # Execute tool call
    call = message.tool_calls[0]
    args = json.loads(call.function.arguments)
    result = get_weather(args["location"])
    
    # Send result back
    final = client.completion(
        model=model,
        messages=[
            {"role": "user", "content": "What is the weather in Paris?"},
            {"role": "assistant", "tool_calls": [call.to_dict()]},
            {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)},
        ],
    )
    
    print(final.choices[0].message.content)
    client.close()

# Development
run_agent("ollama/qwen3:4b", config={})

# Production
run_agent(
    "vllm/gpt-oss-20b",
    config={
        "vllm": {
            "base_url": os.environ["VLLM_BASE_URL"],
            "api_key": os.environ["VLLM_API_KEY"],
        }
    },
)
```

## Checklist

When moving an application from local to production:

1. **Install the right extra.** Gateway deployments need `model-calling[server]`.

2. **Update model names.** Change `ollama/` to `vllm/`, `openai/`, or `openrouter/` as appropriate.

3. **Configure the backend.** Pass `config={"prefix": {"base_url": ..., "api_key": ...}}` to `Client()`, or set `OPENAI_API_KEY` / `OPENROUTER_API_KEY` in the environment for hosted services.

4. **Fix vLLM base URLs.** Include `/v1` in `base_url` for vLLM, OpenAI, and OpenRouter.

5. **Close the client.** Use `async with` / `with` or call `aclose()` / `close()` after async or sync use.

6. **Handle unsupported parameters.** The library raises by default. Pass `drop_params=True` to drop them, or remove parameters the backend cannot honor (Ollama cannot force tool calls; Chat Completions backends cannot take `thinking.budget_tokens`).

7. **Test streaming.** Verify that streams close properly and errors at the start of a stream raise before iteration begins.

8. **Secure credentials.** In OpenShift, mount API keys from Secrets. Never commit them to source.

9. **Check error handling.** Catch `model_calling.APIError` or its subclasses (`BadRequestError`, `AuthenticationError`, `RateLimitError`, `InternalServerError`). The `provider`, `status`, and `body` attributes help with debugging.

10. **Consider the gateway.** For multiple applications, a single gateway configured from its environment can route to many backends. Applications call it with the official `openai` or `anthropic` SDKs.

## Upgrading from 0.1.x

### Install the server extra

The library and gateway dependencies are split. Applications that only call models in-process still install `model-calling`, which brings httpx and pydantic and nothing else. A gateway deployment installs `model-calling[server]`, which adds FastAPI, uvicorn, and python-dotenv.

```bash
# In-process use
pip install model-calling

# Gateway deployment
pip install 'model-calling[server]'
```

### Switch to the in-process Client

The in-process `Client` is the primary API. HTTP calls to a running gateway still work, but the recommended path is to import the library directly or to point the official OpenAI and Anthropic SDKs at the gateway.

**Before (0.1.x):**

```python
from model_calling.client import SyncModelCallingClient

client = SyncModelCallingClient(base_url="http://localhost:8000")
response = client.chat_completion(
    model="ollama/qwen3:4b",
    messages=[{"role": "user", "content": "Hello"}],
)
```

**After (0.2.x, in-process):**

```python
from model_calling import Client

client = Client()
response = client.completion(
    model="ollama/qwen3:4b",
    messages=[{"role": "user", "content": "Hello"}],
)
```

**After (0.2.x, official SDKs pointed at the gateway):**

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="chat",  # an alias from the gateway's model-calling.yaml
    messages=[{"role": "user", "content": "Hello"}],
)
```

The gateway does not check client keys. It uses backend keys from its own environment. Gateway clients name models by alias (see [below](#gateway-clients-name-models-by-alias)); the in-process `Client` still takes `prefix/model` names.

The legacy `ModelCallingClient` and `SyncModelCallingClient` HTTP clients are removed in 0.2.x. Call the gateway with the official `openai` or `anthropic` SDK (as shown above), or use the in-process `Client`.

### Update vLLM base_url

OpenAI-compatible base URLs (vLLM, OpenAI, OpenRouter) are the API root with `/v1` included, matching the `openai` SDK's conventions. The library appends `/chat/completions` or `/embeddings`.

**Before (0.1.x):**

```python
config = {"vllm": {"base_url": "http://vllm-server:8000"}}
```

**After (0.2.x):**

```python
config = {"vllm": {"base_url": "http://vllm-server:8000/v1"}}
```

The Ollama base URL is still the server root (`http://localhost:11434`). The library appends `/api/chat` or `/api/embed` depending on the call.

The gateway's `VLLM_BASE_URL` variable follows the same rule. The 0.1.x container files set it to the server root (`http://localhost:8000`); append `/v1` to any value carried over.

### The library is strict by default

A parameter the backend cannot honor raises `UnsupportedParamsError` (a subclass of `BadRequestError`). Pass `drop_params=True` client-wide or per call to drop unsupported parameters instead.

```python
from model_calling import Client

weather_tool = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
}

client = Client(drop_params=True)  # Drop unsupported params for all calls

# Or per call:
response = client.completion(
    model="ollama/qwen3:4b",
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=[weather_tool],
    tool_choice="required",  # Ollama can't force a tool call
    drop_params=True,
)
```

The gateway still drops by default. Set `MODEL_CALLING_STRICT_PARAMS=true` in its environment to turn on strict mode.

### Callback changes

The `on_stream_chunk` hook now receives internal events (`model_calling.core.StreamEvent` types such as `MessageStart` and `BlockDelta`) instead of raw backend lines. These events are the same shape for every backend, so callback code that switches on event type no longer needs to know which backend answered.

A stream that stops early (the consumer breaks out of the loop and closes the iterator, or cancels the async task) ends with `on_request_end(cancelled=True)` instead of `on_error`.

### Backend 401/403 becomes 502

When a gateway call reaches a backend and the backend refuses the gateway's credential (401 or 403), the gateway client sees 502 (the gateway's fault, not the client's). Other 4xx statuses pass through unchanged.

### vLLM prompt mode is gone

The legacy `/generate` endpoint and the `openai_compatible` adapter switch are removed. vLLM is always reached through its OpenAI-compatible `/v1/chat/completions` and `/v1/embeddings` endpoints.

### Embeddings changes

The `/v1/embeddings` response returns the full model id (as `ollama/nomic-embed-text`) instead of the short name, and includes the backend's token count in the `usage` object when the backend reports one. Ollama embeddings are batched through `/api/embed` rather than sent one at a time to `/api/embeddings`. The gateway's in-memory embedding cache applied only to Cohere in 0.2.0, which no route reached; the Cohere v2 backend added in 0.3.0 doesn't use it.

### Stream error mapping

Errors a backend reports inside a stream now map by kind: rate limit errors become 429, not found becomes 404, invalid request becomes 400. Errors that do not match those kinds still become 502.

### Messages endpoint requires max_tokens

The gateway's `POST /v1/messages` endpoint requires `max_tokens` in the request body. Requests without it return 400.

### Runtime configuration removed

`POST /v1/config/update`, `MODEL_CALLING_ADMIN_TOKEN`, and the legacy client's `update_config` and `admin_token` are removed. Backends are configured in the alias file and environment (see [Model Aliases](advanced.md#model-aliases)).

### Gateway request bodies reject reserved fields

Request bodies sent to the gateway may not contain `api_base`, `api_key`, or `drop_params`. These are reserved for the gateway's internal use (to avoid sending the gateway's credentials elsewhere or overriding its parameter policy). Requests containing any of these fields return 400 with a message listing the rejected fields.

### Gateway responses order JSON keys as the SDK types do

Non-streaming gateway responses list their keys in the order the official `openai` and `anthropic` SDK types define them, which can differ from 0.1.x. Clients that parse JSON are unaffected; anything comparing raw response text may need updating.

### Gateway clients name models by alias

In 0.1.x a gateway client named the backend in the model string (`ollama/qwen3:4b`), so an app changed when its backend did. Now the gateway serves only the aliases defined in its YAML file (`MODEL_CALLING_CONFIG`, default `model-calling.yaml`), and any other name gets 404, including every `prefix/model` name. To migrate, write an alias for each model your apps use. The alias can keep the old string as its name if changing apps is not practical:

```yaml
models:
  "ollama/qwen3:4b": {backend: ollama, model: qwen3:4b}
```

Responses carry the alias in `model`, `GET /v1/models` lists the aliases (it no longer queries Ollama or vLLM, or lists fixed OpenAI and Cohere names), and unprefixed names no longer default to Ollama. Cohere models were not reachable in 0.2.0; from 0.3.0 an alias reaches them through `provider: cohere`, the Cohere v2 backend. See [Model Aliases](advanced.md#model-aliases).

### POST /v1/models/refresh is removed

The gateway no longer probes model capabilities, so `POST /v1/models/refresh` is gone. The 0.1.x Cohere adapters are removed with it; Cohere returns in 0.3.0 as a Cohere v2 backend on the provider interface, with no live coverage.
