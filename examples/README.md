# Model Calling Examples

Example scripts demonstrating the model-calling library and gateway.

## Library examples

Feature examples that use the library directly with prefixed model names (`ollama/qwen3:0.6b`, `vllm/...`, `openai/...`, etc.):

- **library_usage.py** — sync Client with chat completions and tool calling
- **ollama_example.py** — async Client with chat and streaming
- **agent_example.py** — building an agent with tool calling
- **embeddings_example.py** — embeddings and similarity search
- **compare_embeddings.py** — comparing multiple embedding models
- **hosted_providers.py** — OpenAI and Anthropic backends (API keys required)
- **reasoning_example.py** — reasoning models (if available locally)
- **vision_example.py** — vision models for image analysis (if available)
- **vllm_example.py** — vLLM backend (requires a running vLLM server)

Run from the repo root:

```bash
uv run python examples/library_usage.py
```

## Gateway example

**gateway_example.py** drives a running gateway with the official `openai` and `anthropic` SDKs. The gateway resolves model aliases from its config file.

Start the compose stack (mounts `examples/config/model-calling.yaml`):

```bash
podman-compose up
```

Then run the example:

```bash
uv run python examples/gateway_example.py
```

## Configuration

**config/model-calling.yaml** — sample alias file showing local Ollama backend and how to switch to production vLLM.

## Model names

Library examples use prefixed names: `ollama/qwen3:0.6b`, `vllm/mistral-7b`. Gateway examples use aliases defined in the config file: `chat`, `embed`.

Tool calling support depends on the model. Unsupported parameters raise an error by default; pass `drop_params=True` to drop them instead.
