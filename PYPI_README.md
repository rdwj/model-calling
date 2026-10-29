# model-calling

[![PyPI version](https://badge.fury.io/py/model-calling.svg)](https://badge.fury.io/py/model-calling)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

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
| `ollama` | `http://localhost:11434` | none |
| `vllm` | `http://localhost:8000/v1` | `api_key` in config if needed |
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY`, optional `OPENAI_ORG_ID` |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |

The library reads only those environment variables. Override per call with `api_base=` / `api_key=`, or per client via `config`.

Anthropic and Cohere backends are not in the library yet; they are planned. The gateway reaches them through legacy adapters: non-streaming Chat Completions only (streaming returns 501), plus Cohere embeddings.

Both formats work over every backend. You can call `client.messages()` against Ollama, vLLM, OpenAI, or OpenRouter; the library translates to their Chat Completions APIs. Likewise for `client.completion()`.

## Strictness

Strict by default: a parameter the backend cannot honor raises `UnsupportedParamsError` (a `BadRequestError` with a `.params` list) before anything is sent. Pass `drop_params=True` (client-wide or per call) to drop them instead.

Example: Ollama cannot force a tool call (`tool_choice` "required" or `{"type": "any"}`). Chat Completions backends cannot take `thinking.budget_tokens`.

Some parameters are always dropped silently: `cache_control` (a caching hint that changes no output) and `user` for Ollama embeddings.

## Gateway

The gateway is a FastAPI server running on the same library code. Apps talk to it with the official `openai` or `anthropic` SDK and name models by alias; a YAML file decides which backend serves each alias. Moving from Ollama in development to vLLM in production is an edit to that file. The app doesn't change, and the gateway picks up the edit without a restart.

```yaml
# model-calling.yaml (or the file MODEL_CALLING_CONFIG names)
backends:
  prod: {provider: vllm, base_url: https://vllm.example.com/v1, api_key_env: VLLM_API_KEY}
models:
  chat: {backend: ollama, model: qwen3:4b}   # development
  # chat: {backend: prod, model: gpt-oss-20b}   # production
  embed: {backend: ollama, model: nomic-embed-text, dimensions: 768}
```

```bash
model-calling  # or: python -m model_calling
```

Binds `0.0.0.0`, port from `PORT` (default 8000). The gateway reads `.env` (python-dotenv) at startup.

### Endpoints

- `POST /v1/chat/completions` (OpenAI Chat Completions; streaming SSE ending `data: [DONE]`)
- `POST /v1/messages` (Anthropic Messages; streaming SSE with `event:` lines; errors as Anthropic error bodies)
- `POST /v1/embeddings` (OpenAI Embeddings)
- `GET /v1/models` (the configured aliases), `POST /v1/moderations`, `GET /` health

Point the official SDKs at it:

```python
import openai
import anthropic

openai_client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
anthropic_client = anthropic.Anthropic(base_url="http://localhost:8000", api_key="unused")

reply = openai_client.chat.completions.create(
    model="chat", messages=[{"role": "user", "content": "Hello"}]
)
```

The gateway does not check client keys; it uses its own backend keys from its environment. Any name that isn't an alias, including a backend-prefixed one such as `ollama/qwen3:4b`, gets 404. See [Model Aliases](https://github.com/rdwj/model-calling/blob/main/docs/advanced.md#model-aliases) for the file format and reload behavior.

### Strict mode

The gateway drops unsupported parameters by default. Set `MODEL_CALLING_STRICT_PARAMS=true` (accepts true/false/1/0/yes/no) to turn on strict mode: 400 with message "model '\<alias\>' cannot honor \<param\>".

### Backend URLs

For backends the alias file names without a `base_url`, the gateway takes URLs from its environment: `OLLAMA_BASE_URL` is Ollama's native root (`http://ollama:11434`), while `VLLM_BASE_URL` and `MODEL_CALLING_ANTHROPIC_BASE_URL` are API roots with `/v1`. `VLLM_API_KEY` is sent to vLLM if set.

Environment: `MODEL_CALLING_CONFIG`, `OPENAI_API_KEY`, `OPENAI_ORG_ID`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `OLLAMA_BASE_URL`, `VLLM_BASE_URL`, `VLLM_API_KEY`, `MODEL_CALLING_ANTHROPIC_BASE_URL`, `PORT`, `MODEL_CALLING_STRICT_PARAMS`. The `model-calling` command also reads a `.env` file from the directory it starts in.

## Documentation

For complete documentation, visit the [model-calling documentation](https://github.com/rdwj/model-calling/tree/main/docs).

## Examples

See the [examples directory](https://github.com/rdwj/model-calling/tree/main/examples) for more code samples.

## Testing

```bash
uv sync
uv run pytest                 # offline suite; no network
uv run pytest -m integration  # needs a running gateway backed by Ollama
uv run pytest -m live         # one cheap OpenRouter call per format
```

The offline suite runs the gateway in-process against recorded backend fixtures and validates output with the official `openai` and `anthropic` SDK types. See [tests/README.md](https://github.com/rdwj/model-calling/blob/main/tests/README.md) for the harness, fixture recorder, and integration scripts.

## License

Apache-2.0. See [LICENSE](https://github.com/rdwj/model-calling/blob/main/LICENSE) and [NOTICE](https://github.com/rdwj/model-calling/blob/main/NOTICE).

## Acknowledgements

This project was developed with assistance from AI tools. It was inspired by the need for a flexible approach to building LLM applications with locally-hosted and cloud models.
