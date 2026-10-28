# model-calling documentation

An LLM client library for Python that speaks OpenAI Chat Completions and Anthropic Messages over Ollama, vLLM, OpenAI-compatible, and OpenRouter backends.

## Install

```bash
pip install model-calling              # library only
pip install 'model-calling[server]'    # adds the gateway (FastAPI + uvicorn)
pip install 'model-calling[moderation]' # adds moderation endpoint (torch + transformers)
```

## Quick example

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

client.close()
```

## Contents

- [Quickstart](quickstart.md): Get up and running
- [API Reference](api-reference.md): Comprehensive API documentation
- [Examples](examples.md): Code examples for common use cases
- [Advanced Topics](advanced.md): In-depth information
- [Migration Guide](migration-guide.md): Migrating from 0.1.x
- [Testing Guide](testing.md): Running the test suite
