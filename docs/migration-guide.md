# Migration Guide

This guide covers upgrading from model-calling 0.1.x and moving applications from local development to production.

## Contents

- [Upgrading from 0.1.x](#upgrading-from-01x)
- [From Local Development to Production](#from-local-development-to-production)
- [Checklist](#checklist)

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
    model="ollama/qwen3:4b",
    messages=[{"role": "user", "content": "Hello"}],
)
```

The gateway does not check client keys. It uses backend keys from its own environment.

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

The `on_stream_chunk` hook now receives internal events (`model_calling.core.StreamEvent`: `MessageStart`, `BlockDelta`, etc.) instead of raw backend lines. These events are the same shape for every backend, so callback code that switches on event type no longer needs to know which backend answered.

A stream that stops early (the consumer breaks out of the loop and closes the iterator, or cancels the async task) ends with `on_request_end(cancelled=True)` instead of `on_error`.

### Backend 401/403 becomes 502

When a gateway call reaches a backend and the backend refuses the gateway's credential (401 or 403), the gateway client sees 502 (the gateway's fault, not the client's). Other 4xx statuses pass through unchanged.

### vLLM prompt mode is gone

The legacy `/generate` endpoint and the `openai_compatible` adapter switch are removed. vLLM is always reached through its OpenAI-compatible `/v1/chat/completions` and `/v1/embeddings` endpoints.

### Embeddings changes

The `/v1/embeddings` response returns the full model id (as `ollama/nomic-embed-text`) instead of the short name, and includes the backend's token count in the `usage` object when the backend reports one. Ollama embeddings are batched through `/api/embed` rather than sent one at a time to `/api/embeddings`. The gateway's in-memory embedding cache now applies only to Cohere (the last unported backend that reaches it).

### Stream error mapping

Errors a backend reports inside a stream now map by kind: rate limit errors become 429, not found becomes 404, invalid request becomes 400. Errors that do not match those kinds still become 502.

### Messages endpoint requires max_tokens

The gateway's `POST /v1/messages` endpoint requires `max_tokens` in the request body. Requests without it return 400.

## From Local Development to Production

A typical workflow starts with Ollama for local iteration, moves to a hosted service (OpenAI, OpenRouter) or vLLM on OpenShift for shared testing, and deploys to production with environment-specific model names and authentication.

The library reads only backend API keys (`OPENAI_API_KEY`, `OPENAI_ORG_ID`, `OPENROUTER_API_KEY`) from the environment. It never reads a `.env` file. The gateway loads `.env` with python-dotenv at startup. Both the library and the gateway support per-prefix configuration with `base_url`, `api_key`, and `headers`.

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

# Read your app's own config (the library itself only reads API key env vars)
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

Applications that run in dev, test, and prod can read their own environment variables and build the client config from them. The library itself only reads the provider API key variables listed above.

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

A gateway serves many applications. Each app points the `openai` or `anthropic` SDK at the gateway, and the gateway holds the backend configurations and keys. Update a backend URL with `POST /v1/config/update`:

```bash
curl -X POST http://localhost:8000/v1/config/update \
  -H 'Content-Type: application/json' \
  -d '{"prefix": "vllm", "config": {"base_url": "https://vllm.apps.cluster.example.com/v1"}}'
```

The gateway reads backend keys from its environment (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`). In OpenShift, mount them from Secrets:

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
    model="openrouter/mistralai/mistral-nemo",
    messages=[{"role": "user", "content": "Hello"}],
)
```

The same gateway serves Chat Completions (`POST /v1/chat/completions`), Anthropic Messages (`POST /v1/messages`), and Embeddings (`POST /v1/embeddings`).

### Anthropic Messages over any backend

The `messages` method translates Anthropic Messages requests to the backend's format. This works over Ollama, vLLM, OpenAI, and OpenRouter. The Anthropic and Cohere backends are not in the library yet (planned).

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

10. **Consider the gateway.** For multiple applications, a single gateway with `/v1/config/update` can route to many backends. Applications call it with the official `openai` or `anthropic` SDKs.
