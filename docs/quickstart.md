# Quickstart

This guide gets you started with model-calling, a library and gateway for calling LLM providers through a consistent API. The library supports OpenAI Chat Completions and Anthropic Messages side by side, over Ollama, vLLM, OpenAI, and OpenRouter backends.

## Installation

Install the library:

```bash
pip install model-calling
```

The library depends on httpx and pydantic. It never loads FastAPI and never reads a `.env` file. For the optional gateway server, install the `server` extra:

```bash
pip install 'model-calling[server]'
```

This adds FastAPI, uvicorn, and python-dotenv.

## First call

Pull a model with Ollama:

```bash
ollama pull qwen3:4b
```

Call it with the Chat Completions API:

```python
from model_calling import completion

response = completion(
    model="ollama/qwen3:4b",
    messages=[{"role": "user", "content": "What is prompt caching?"}],
)

print(response.choices[0].message.content)
```

The response is a pydantic model shaped like `openai.types.chat.ChatCompletion`. Access fields as attributes, or call `.to_dict()` for the wire representation. Unprefixed model names default to Ollama, so `"qwen3:4b"` works the same way.

## Anthropic Messages

The same backend answers Messages requests:

```python
from model_calling import messages

response = messages(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "What is prompt caching?"}],
    max_tokens=512,
)

text = next(b.text for b in response.content if b.type == "text")
print(text)
```

The response validates against `anthropic.types.Message`. Both formats work over every backend. Ollama, vLLM, OpenAI, and OpenRouter all serve Chat Completions and Messages requests through translation.

## Streaming

Request a stream with `stream=True`:

```python
from model_calling import completion

stream = completion(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "Write a haiku about recursion."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
print()
```

Each chunk is a `ChatCompletionChunk`. The stream returns once the backend has accepted the request, so a refusal raises before you start iterating.

Messages streams work the same way:

```python
from model_calling import messages

stream = messages(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "Explain extended thinking."}],
    max_tokens=512,
    stream=True,
)

for event in stream:
    if event.type == "content_block_delta" and event.delta.type == "text_delta":
        print(event.delta.text, end="", flush=True)
print()
```

The events are `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, and `message_stop`, matching the Anthropic stream API.

## Tool calling

Define tools as OpenAI function schemas and pass them to `completion`:

```python
from model_calling import completion
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_temperature",
            "description": "Get the current temperature for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    }
]

response = completion(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "What's the temperature in Paris?"}],
    tools=tools,
)

message = response.choices[0].message
if message.tool_calls:
    call = message.tool_calls[0]
    print(f"Tool: {call.function.name}")
    print(f"Arguments: {call.function.arguments}")
    
    # Simulate calling your function
    result = {"temperature": 18, "unit": "celsius"}
    
    # Continue the conversation with the result
    final = completion(
        model="qwen3:4b",
        messages=[
            {"role": "user", "content": "What's the temperature in Paris?"},
            message.to_dict(),
            {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)},
        ],
        tools=tools,
    )
    
    print(final.choices[0].message.content)
```

Not every backend supports forcing a tool call. Ollama rejects `tool_choice` set to `"required"` or `{"type": "any"}`. With strict mode (the library's default), an unsupported parameter raises `UnsupportedParamsError`. Pass `drop_params=True` to send the request without it instead.

## Embeddings

Pull an embedding model:

```bash
ollama pull nomic-embed-text
```

Call it with the Embeddings API:

```python
from model_calling import embedding

response = embedding(
    model="ollama/nomic-embed-text",
    input=["prompt caching", "extended thinking"],
)

for item in response.data:
    print(f"Embedding {item.index}: {len(item.embedding)} dimensions")
```

The response is a `CreateEmbeddingResponse` matching the OpenAI SDK. Ollama accepts text input only; token arrays raise `BadRequestError`. The OpenAI-compatible backends pass token arrays through to the server.

## Switching backends

Model names are `prefix/name`. Override the default base URL or API key per call:

```python
from model_calling import completion

# vLLM (local or cluster)
response = completion(
    model="vllm/gpt-oss-20b",
    messages=[{"role": "user", "content": "Hello"}],
    api_base="http://cluster.local:8000/v1",
)

# OpenAI (reads OPENAI_API_KEY from the environment)
response = completion(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
)

# OpenRouter (reads OPENROUTER_API_KEY from the environment)
response = completion(
    model="openrouter/mistralai/mistral-nemo",
    messages=[{"role": "user", "content": "Hello"}],
)
```

The library reads `OPENAI_API_KEY`, `OPENAI_ORG_ID`, and `OPENROUTER_API_KEY` from the environment. It never reads a `.env` file. For per-call credentials, pass `api_key`:

```python
response = completion(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    api_key="sk-...",
)
```

The default base URLs are `http://localhost:11434` for Ollama, `http://localhost:8000/v1` for vLLM, `https://api.openai.com/v1` for OpenAI, and `https://openrouter.ai/api/v1` for OpenRouter. Override them with `api_base` or configure a client (see the [API reference](api-reference.md)).

## Using a Client

The module-level functions (`completion`, `messages`, `embedding`) run on a lazily created default client. For custom config, callbacks, or control over when connections close, create your own:

```python
from model_calling import Client

client = Client(
    config={
        "ollama": {"base_url": "http://192.168.1.100:11434"},
        "openai": {"api_key": "sk-..."},
    },
    drop_params=True,
)

response = client.completion(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "Hello"}],
)

client.close()
```

Or use a context manager:

```python
with Client() as client:
    response = client.completion(
        model="qwen3:4b",
        messages=[{"role": "user", "content": "Hello"}],
    )
```

For async calls, use `acompletion`, `amessages`, and `aembedding`, and close with `aclose()` or `async with`.

## Running the gateway

Install the server extra if you have not already:

```bash
pip install 'model-calling[server]'
```

Apps name gateway models by alias. Create `model-calling.yaml` in the directory you start the gateway from, mapping each alias to a backend and model:

```yaml
models:
  chat: {backend: ollama, model: qwen3:4b}
```

When you move to production, point `chat` at another backend in this file; the app keeps asking for `chat`, and the gateway picks up the edit without a restart. See [Model Aliases](advanced.md#model-aliases).

Start the gateway:

```bash
model-calling
```

Or:

```bash
python -m model_calling
```

The gateway listens on port 8000 (override with the `PORT` environment variable). It reads `.env` at startup for backend credentials.

Point the official SDKs at it:

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="chat",
    messages=[{"role": "user", "content": "Hello"}],
)

print(response.choices[0].message.content)
```

Or with the Anthropic SDK:

```python
import anthropic

client = anthropic.Anthropic(base_url="http://localhost:8000", api_key="unused")

response = client.messages.create(
    model="chat",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=512,
)

print(next(b.text for b in response.content if b.type == "text"))
```

The gateway does not check client API keys. It uses its own backend credentials from the environment. Requests can also use `curl`:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chat",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

The gateway serves `POST /v1/chat/completions`, `POST /v1/messages`, `POST /v1/embeddings`, `GET /v1/models`, and `POST /v1/config/update` (off unless an admin token is set). See the [API reference](api-reference.md) for details.

## Next steps

Read the [API reference](api-reference.md) for the complete Client API, callback hooks, error types, and gateway endpoints.
