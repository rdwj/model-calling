# Model Calling Documentation

Welcome to the Model Calling documentation. This guide will help you use the unified API for calling multiple LLM providers.

## Contents

- [API Reference](api-reference.md) - Comprehensive API documentation
- [Quickstart](quickstart.md) - Get up and running quickly
- [Examples](examples.md) - Code examples for common use cases
- [Migration Guide](migration-guide.md) - Transitioning from local development to production
- [Testing Guide](testing.md) - Testing the service with local models
- [Advanced Topics](advanced.md) - In-depth information for advanced users

## What is Model Calling?

Model Calling is a unified API that allows you to interact with multiple Large Language Models (LLMs) through a consistent interface. Whether you're using local models via Ollama, cluster-deployed models with vLLM, or hosted models from providers like OpenAI and Anthropic, Model Calling provides a single, OpenAI-compatible API.

### Key Features

- **Unified API**: Call any supported model using the same code
- **Function Calling**: Tool/function calling support across models  
- **Streaming Support**: Efficient streaming responses
- **Multiple Providers**: Support for Ollama, vLLM, OpenAI, Anthropic, Cohere, and more
- **Python Client**: Easy-to-use synchronous and asynchronous clients
- **Runtime Configuration**: Update model settings without restarting

### Supported Providers

| Provider | Description | Example Models |
|----------|-------------|----------------|
| Ollama | Local models | mistral-small3.1, llama3, qwen |
| vLLM | Cluster-deployed models | Any model deployed with vLLM |
| OpenAI | OpenAI API | gpt-4, gpt-3.5-turbo |
| Anthropic | Anthropic API | claude-3-opus, claude-3-sonnet |
| Cohere | Cohere API | command, command-r |

## Installation

```bash
pip install model-calling
```

## Quick Example

```python
from model_calling.client import SyncModelCallingClient

client = SyncModelCallingClient()

try:
    response = client.chat_completion(
        model="ollama/mistral-small3.1:24b",
        messages=[
            {"role": "user", "content": "What is quantum computing?"}
        ]
    )
    
    print(response["choices"][0]["message"]["content"])
finally:
    client.close()
```

## Next Steps

- Read the [API Reference](api-reference.md) for comprehensive documentation
- See the [Quickstart](quickstart.md) guide to get up and running
- Explore [Examples](examples.md) for code samples of common patterns
- Check out [Advanced Topics](advanced.md) for in-depth information
