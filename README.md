# model-calling

An LLM client library for Python that speaks OpenAI Chat Completions and Anthropic Messages over Ollama, vLLM, OpenAI-compatible, and OpenRouter backends. Use it in-process or as a gateway.

## Install

```bash
pip install model-calling              # library only
pip install 'model-calling[server]'    # adds the gateway (FastAPI + uvicorn)
pip install 'model-calling[moderation]' # adds moderation endpoint (torch + transformers)
```

The library depends on httpx and pydantic. Importing it never loads FastAPI and never reads `.env`.

## Quick start

```python
from model_calling import Client

client = Client()

# Chat Completions (OpenAI format)
response = client.completion(
    "ollama/qwen3:4b",
    [{"role": "user", "content": "What is the capital of France?"}]
)
print(response.choices[0].message.content)

# Anthropic Messages
message = client.messages(
    "vllm/gpt-oss-20b",
    [{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=256
)
# Reasoning models put a thinking block first; print the text.
print(next(b.text for b in message.content if b.type == "text"))

# Embeddings
vectors = client.embedding("ollama/nomic-embed-text", ["hello", "world"])
print(vectors.data[0].embedding[:5])

client.close()
```

### Streaming

```python
from model_calling import Client

client = Client()

# Chat Completions stream
for chunk in client.completion(
    "ollama/qwen3:4b",
    [{"role": "user", "content": "Count to five."}],
    stream=True
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

# Messages stream
for event in client.messages(
    "vllm/gpt-oss-20b",
    [{"role": "user", "content": "Count to five."}],
    max_tokens=256,
    stream=True
):
    if event.type == "content_block_delta" and event.delta.type == "text_delta":
        print(event.delta.text, end="", flush=True)

client.close()
```

### Async

```python
from model_calling import Client
import asyncio

async def main():
    async with Client() as client:
        response = await client.acompletion(
            "ollama/qwen3:4b",
            [{"role": "user", "content": "Hi!"}]
        )
        print(response.choices[0].message.content)

asyncio.run(main())
```

Module-level functions are also available:

```python
import model_calling

response = model_calling.completion("ollama/qwen3:4b", [...])
vectors = model_calling.embedding("ollama/nomic-embed-text", ["hello"])
```

## Backends

Model routing uses `prefix/name`. Unprefixed names go to Ollama.

| Prefix | Default base_url | Credentials |
|--------|------------------|-------------|
| `anthropic` | `https://api.anthropic.com/v1` | `ANTHROPIC_API_KEY` |
| `ollama` | `http://localhost:11434` | none |
| `vllm` | `http://localhost:8000/v1` | `api_key` in config if needed |
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY`, optional `OPENAI_ORG_ID` |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |

The library reads only those environment variables. Override per call with `api_base=` / `api_key=`, or per client via `config`.

The `anthropic` prefix speaks Anthropic Messages to any server that serves it at `{base_url}/messages`: Anthropic, OpenRouter (`https://openrouter.ai/api/v1`), or vLLM (`http://localhost:8000/v1`). A `messages()` call reaches it unchanged. A `completion()` call is translated, and since Messages requires `max_tokens`, one sent without it gets 4096.

The Cohere backend is not in the library yet; it is planned. The gateway reaches it through a legacy adapter: non-streaming Chat Completions only (streaming returns 501), plus embeddings.

Both formats work over every backend. You can call `client.messages()` against Ollama, vLLM, OpenAI, or OpenRouter; the library translates to their Chat Completions APIs. Likewise for `client.completion()` against an Anthropic backend.

## Strictness

Strict by default: a parameter the backend cannot honor raises `UnsupportedParamsError` (a `BadRequestError` with a `.params` list) before anything is sent. Pass `drop_params=True` (client-wide or per call) to drop them instead.

Example: Ollama cannot force a tool call (`tool_choice` "required" or `{"type": "any"}`). Chat Completions backends cannot take `thinking.budget_tokens`.

Some parameters are always dropped silently: `cache_control` (a caching hint that changes no output) and `user` for Ollama embeddings.

## Gateway

The gateway is a FastAPI server running on the same library code.

```bash
model-calling  # or: python -m model_calling
```

Binds `0.0.0.0`, port from `PORT` (default 8000). The gateway reads `.env` (python-dotenv) at startup.

### Endpoints

- `POST /v1/chat/completions` (OpenAI Chat Completions; streaming SSE ending `data: [DONE]`)
- `POST /v1/messages` (Anthropic Messages; streaming SSE with `event:` lines; errors as Anthropic error bodies)
- `POST /v1/messages/count_tokens` (Anthropic token counting, answered with a local estimate since no backend here has a counter)
- `POST /v1/embeddings` (OpenAI Embeddings)
- `GET /v1/models`, `POST /v1/models/refresh`, `POST /v1/config/update`, `POST /v1/moderations`, `GET /` health

Point the official SDKs at it:

```python
import openai
import anthropic

openai_client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
anthropic_client = anthropic.Anthropic(base_url="http://localhost:8000", api_key="unused")
```

The gateway does not check client keys; it uses its own backend keys from its environment.

### Strict mode

The gateway drops unsupported parameters by default. Set `MODEL_CALLING_STRICT_PARAMS=true` (accepts true/false/1/0/yes/no) to turn on strict mode: 400 with message "\<provider\> cannot honor \<param\>".

Set a backend's URL at runtime with `POST /v1/config/update`, for example `{"prefix": "ollama", "config": {"base_url": "http://ollama:11434"}}`. The `OLLAMA_BASE_URL` and `VLLM_BASE_URL` variables the container files set are logged at startup but not applied yet. This endpoint is unauthenticated, so keep the gateway on a trusted network.

Environment: `OPENAI_API_KEY`, `OPENAI_ORG_ID`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `COHERE_API_KEY`, `PORT`, `MODEL_CALLING_STRICT_PARAMS`.

## Containers

Build with podman:

```bash
./run_podman.sh         # Containerfile directly
./run_podman_compose.sh # podman-compose.yml
```

UBI 9 Python 3.12 base, runs as non-root. FIPS variant in `Containerfile-FIPS`.

## Testing

```bash
uv sync
uv run pytest                 # offline suite; no network
uv run pytest -m integration  # needs a running gateway backed by Ollama
uv run pytest -m live         # one cheap OpenRouter call per format
```

The offline suite runs the gateway in-process against recorded backend fixtures and validates output with the official `openai` and `anthropic` SDK types. See [tests/README.md](tests/README.md) for the harness, fixture recorder, and integration scripts.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Acknowledgements

This project was developed with assistance from AI tools. It was inspired by the need for a flexible approach to building LLM applications with locally-hosted and cloud models.
