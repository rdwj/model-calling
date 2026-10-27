# Model Calling Service

A unified API service that provides an OpenAI-compatible interface to various LLM backends, including Ollama and vLLM. This service allows you to use multiple model types through a consistent API, making it easy to build applications that can switch between different models.

## Key Features

- 🔄 **OpenAI-compatible API**: Uses the familiar chat completions format
- ☎️ **Multiple Backends**: Support for Ollama, vLLM, OpenAI, Anthropic, Cohere, and more
- 🛠️ **Function Calling**: Unified support for tools/function calling across models
- 📊 **Streaming Support**: Efficient streaming for all supported models
- 🔧 **Runtime Configuration**: Adjust model settings without restarting
- 📦 **Importable Library**: Can be used as a service or imported library
- 🎯 **Callback System**: Extensible architecture for monitoring and customization

## Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) for local models
- [vLLM](https://github.com/vllm-project/vllm) (optional, for serving models on GPU clusters)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/rdwj/model-calling.git
cd model-calling
```

2. Install the locked environment with [uv](https://docs.astral.sh/uv/):

```bash
# Core package plus development tools, at the exact versions in uv.lock
uv sync --frozen

# Optional: example dependencies
uv sync --frozen --extra examples
```

model-calling pins every dependency to an exact, vetted version (see
"Dependency Policy" in [CONTRIBUTING.md](CONTRIBUTING.md)). If you install it
into your own project, lock your environment too, and consider a release-age
cutoff such as uv's `exclude-newer` so a freshly compromised release can't
reach you before the ecosystem catches it.

## Quick Start

1. **Ensure you have embedding models installed in Ollama**:

```bash
# Install embedding models
ollama pull nomic-embed-text:v1.5
ollama pull mxbai-embed-large:335m
```

2. **Start the service**:

```bash
# Make script executable
chmod +x start_service.sh

# Start the service
./start_service.sh
```

This will start the service on http://localhost:8000.

2. **Test with an example script**:

```bash
python -m examples.ollama_example
```

3. **Make API calls directly**:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/mistral-small3.1:24b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

## Hosted Provider Support

In addition to local models, this service supports major hosted LLM providers:

### Available Providers

- **OpenAI**: Access GPT models through the OpenAI API
- **Anthropic**: Access Claude models through the Anthropic API
- **Cohere**: Access Command models through the Cohere API

### Using Hosted Providers

1. Create a `.env` file with your API keys (copy from `.env.example`)

```bash
cp .env.example .env
# Edit .env with your API keys
```

2. Make calls using the provider-specific prefix:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4-turbo",
    "messages": [
      {"role": "user", "content": "What is quantum computing?"}
    ]
  }'
```

3. Or use the client library:

```python
from model_calling.client import SyncModelCallingClient

client = SyncModelCallingClient()
response = client.chat_completion(
    model="anthropic/claude-3-sonnet-20240229",
    messages=[
        {"role": "user", "content": "What is quantum computing?"}
    ]
)
print(response["choices"][0]["message"]["content"])
client.close()
```

Check out `examples/hosted_providers.py` for more examples.

## Containerization

The service ships with a `Containerfile` based on Red Hat UBI images:

#### Option 1: Using Containerfile directly

```bash
# Make the script executable
chmod +x run_podman.sh

# Build and run with Podman
./run_podman.sh
```

#### Option 2: Using podman-compose

```bash
# Make the script executable
chmod +x run_podman_compose.sh

# Run with podman-compose
./run_podman_compose.sh
```

The image uses the Red Hat UBI 9 Python 3.12 base and runs as a non-root user.

## API Reference

### GET /v1/models

List available models.

### POST /v1/chat/completions

Create a chat completion. This endpoint is compatible with OpenAI's API format.

**Request Format**:

```json
{
  "model": "ollama/mistral-small3.1:24b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "stream": false,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather in a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state"
            }
          },
          "required": ["location"]
        }
      }
    }
  ]
}
```

**Response Format**:

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1716322276,
  "model": "ollama/mistral-small3.1:24b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ]
}
```

### POST /v1/config/update

Update configuration for a model type.

**Request Format**:

```json
{
  "prefix": "ollama",
  "config": {
    "base_url": "http://localhost:11434",
    "supports_function_call": true
  }
}
```

## Using as a Library

You can also use model-calling as a library in your own applications:

```python
from model_calling.client import SyncModelCallingClient

client = SyncModelCallingClient()

try:
    response = client.chat_completion(
        model="ollama/mistral-small3.1:24b",
        messages=[
            {"role": "user", "content": "Hello!"}
        ]
    )
    
    print(response["choices"][0]["message"]["content"])
finally:
    client.close()
```

See the examples directory for more detailed usage examples.

## Extending

### Adding New Model Adapters

To add support for new model types, create a new adapter class that extends `ModelAdapter` and register it with the registry:

```python
from model_calling.adapter import ModelAdapter
from model_calling.registry import registry

class MyCustomAdapter(ModelAdapter):
    # Implement required methods
    
# Register the adapter
registry.register_adapter_class("custom", MyCustomAdapter)
```

## Testing

Model Calling includes a comprehensive test suite focusing on:

- Basic functionality with different models
- Function calling (tool use) capabilities 
- Streaming responses
- Error handling

To run the tests:

```bash
# Run all tests
./run_tests.sh

# Run only function calling tests
./test_ollama.sh function

# Run a quick verification
./test_ollama.sh quick

# Auto-configure tests based on available Ollama models
./auto_config_tests.sh

# Complete test setup and auto-configuration
./setup_tests.sh

# Diagnose model compatibility issues
./diagnose_models.sh
```

The tests automatically detect which models are available and will test appropriate functionality.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Acknowledgements

This project was inspired by the need for a flexible, model-agnostic approach to building LLM applications, particularly when working with locally-hosted models like those supported by Ollama and vLLM.

This project was developed with assistance from AI tools.
