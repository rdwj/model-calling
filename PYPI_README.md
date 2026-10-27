# Model Calling

[![PyPI version](https://badge.fury.io/py/model-calling.svg)](https://badge.fury.io/py/model-calling)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A unified API for calling multiple LLM providers through a consistent, OpenAI-compatible interface.

## Key Features

- 🔄 **OpenAI-compatible API**: Uses the familiar chat completions format
- ☎️ **Multiple Backends**: Support for Ollama, vLLM, OpenAI, Anthropic, Cohere, and more
- 🛠️ **Function Calling**: Unified support for tools/function calling across models
- 📊 **Streaming Support**: Efficient streaming for all supported models
- 🔧 **Runtime Configuration**: Adjust model settings without restarting
- 📦 **Importable Library**: Can be used as a service or imported library

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
        model="ollama/mistral-small3.1:24b",  # Use any model from any provider
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning?"}
        ]
    )
    
    print(response["choices"][0]["message"]["content"])
finally:
    client.close()
```

## Supported Providers

| Provider | Prefix | Example Models |
|----------|--------|----------------|
| Ollama (local) | ollama/ | mistral-small3.1, llama3, qwen |
| vLLM (cluster) | vllm/ | Any model deployed with vLLM |
| OpenAI | openai/ | gpt-4, gpt-3.5-turbo |
| Anthropic | anthropic/ | claude-3-opus, claude-3-sonnet |
| Cohere | cohere/ | command, command-r |

## Function Calling

Model Calling provides a consistent interface for function calling (tools) across all supported providers:

```python
import json
from model_calling.client import SyncModelCallingClient

client = SyncModelCallingClient()

tools = [
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
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

try:
    # Initial request with tools
    response = client.chat_completion(
        model="ollama/mistral-small3.1:24b",
        messages=[
            {"role": "user", "content": "What's the weather like in Paris?"}
        ],
        tools=tools
    )
    
    # Check if function call was requested
    message = response["choices"][0]["message"]
    if "function_call" in message:
        function_name = message["function_call"]["name"]
        arguments = json.loads(message["function_call"]["arguments"])
        
        # Call your function with the arguments
        weather_data = get_weather(arguments["location"])
        
        # Continue the conversation with the function result
        final_response = client.chat_completion(
            model="ollama/mistral-small3.1:24b",
            messages=[
                {"role": "user", "content": "What's the weather like in Paris?"},
                {
                    "role": "assistant", 
                    "content": "", 
                    "function_call": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "Paris, France"})
                    }
                },
                {
                    "role": "function", 
                    "name": "get_weather", 
                    "content": json.dumps(weather_data)
                }
            ]
        )
        
        print(final_response["choices"][0]["message"]["content"])
    else:
        print(message["content"])
finally:
    client.close()
```

## Using as a Service

Model Calling can be run as a service to provide a unified API for all your applications:

```bash
# Start the service
python -m model_calling
```

Then make API calls to the service:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/mistral-small3.1:24b",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ]
  }'
```

## Using Hosted Providers

To use hosted providers like OpenAI and Anthropic, set your API keys in environment variables or a .env file:

```bash
# Create a .env file with your API keys
cp .env.example .env
# Edit .env with your API keys
```

Then you can use the hosted models:

```python
from model_calling.client import SyncModelCallingClient

client = SyncModelCallingClient()

try:
    # OpenAI
    response = client.chat_completion(
        model="openai/gpt-4",
        messages=[
            {"role": "user", "content": "What is quantum computing?"}
        ]
    )
    
    # Anthropic
    response = client.chat_completion(
        model="anthropic/claude-3-sonnet-20240229",
        messages=[
            {"role": "user", "content": "What is quantum computing?"}
        ]
    )
finally:
    client.close()
```

## Documentation

For complete documentation, visit the [Model Calling Documentation](https://github.com/rdwj/model-calling/tree/main/docs).

## Examples

Check out the [examples directory](https://github.com/rdwj/model-calling/tree/main/examples) for more examples of how to use Model Calling, including:

- Basic chat completions
- Function calling
- Streaming responses
- Building agents
- Working with different providers

## License

Apache License 2.0. See [LICENSE](https://github.com/rdwj/model-calling/blob/main/LICENSE).
