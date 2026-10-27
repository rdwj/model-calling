# Model Calling API Documentation

A unified API for calling multiple LLM providers through a consistent, OpenAI-compatible interface.

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
  - [Chat Completions](#chat-completions)
  - [Models](#models)
  - [Configuration](#configuration)
- [Tools & Function Calling](#tools--function-calling)
- [Working with Different Providers](#working-with-different-providers)
- [Client Libraries](#client-libraries)
- [Monitoring & Observability](#monitoring--observability)
- [Examples](#examples)

## Introduction

Model Calling provides a unified API for working with multiple large language models (LLMs) through a consistent interface. This allows developers to build applications that can easily switch between different models and providers without changing their code.

The API follows the OpenAI API format, making it familiar to developers already working with LLMs, while extending it to support multiple providers including:

- Local models via Ollama
- Cluster-deployed models via vLLM
- Hosted providers: OpenAI, Anthropic, Cohere, and more

Key features include:
- Consistent interface for all models
- Tool/function calling across supported models
- Streaming support
- Runtime configuration
- Client libraries for Python

## Getting Started

### Installation

```bash
pip install model-calling
```

### Starting the Service

```bash
# Start the service on http://localhost:8000
python -m model_calling
```

### Basic Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "ollama/mistral-small3.1:24b",
        "messages": [
            {"role": "user", "content": "What is machine learning?"}
        ]
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

## API Reference

### Chat Completions

The primary endpoint for generating text from models.

```
POST /v1/chat/completions
```

#### Request Body

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| model | string | Yes | Model identifier in format `provider/model-name` |
| messages | array | Yes | Array of message objects with `role` and `content` |
| temperature | number | No | Sampling temperature (0-2), defaults to 1 |
| top_p | number | No | Nucleus sampling parameter (0-1), defaults to 1 |
| n | integer | No | Number of completions to generate, defaults to 1 |
| stream | boolean | No | Whether to stream responses, defaults to false |
| max_tokens | integer | No | Maximum tokens to generate, provider-dependent |
| tools | array | No | Array of tool objects the model may use |
| stop | string/array | No | Sequence(s) where the model should stop generating |

##### Message Object

| Field | Type | Description |
|-------|------|-------------|
| role | string | Role of the message sender: "system", "user", "assistant", or "tool" |
| content | string | Content of the message |
| name | string | Optional name for tool messages |
| function_call | object | Optional function call information for assistant messages |

##### Tool Object

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get weather information for a location",
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
```

#### Response Object

```json
{
  "id": "chatcmpl-123abc",
  "object": "chat.completion",
  "created": 1694268762,
  "model": "ollama/mistral-small3.1:24b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Machine learning is a subset of artificial intelligence..."
      },
      "finish_reason": "stop"
    }
  ]
}
```

For function calls, the response includes a `function_call` property in the message:

```json
{
  "id": "chatcmpl-123abc",
  "object": "chat.completion",
  "created": 1694268762,
  "model": "ollama/mistral-small3.1:24b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "",
        "function_call": {
          "name": "get_weather",
          "arguments": "{\"location\":\"San Francisco, CA\"}"
        }
      },
      "finish_reason": "function_call"
    }
  ]
}
```

### Models

List available models.

```
GET /v1/models
```

#### Response

```json
{
  "object": "list",
  "data": [
    {"id": "ollama/mistral-small3.1:24b", "object": "model"},
    {"id": "ollama/llama3:8b", "object": "model"},
    {"id": "openai/gpt-4", "object": "model"},
    {"id": "anthropic/claude-3-opus-20240229", "object": "model"}
  ]
}
```

### Configuration

Update configuration for a model type.

```
POST /v1/config/update
```

#### Request Body

```json
{
  "prefix": "ollama",
  "config": {
    "base_url": "http://localhost:11434",
    "supports_function_call": true
  }
}
```

## Tools & Function Calling

Model Calling supports OpenAI-compatible function calling (also known as tools) across multiple providers. This allows the model to request that your application call specific functions.

### Basic Example

#### 1. Define your tools

```json
"tools": [
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get current weather in a location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "City name"
          }
        },
        "required": ["location"]
      }
    }
  }
]
```

#### 2. Send a request that might need function calling

```python
response = client.chat_completion(
    model="ollama/mistral-small3.1:24b",
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"}
    ],
    tools=tools
)
```

#### 3. Check if a function call was requested

```python
message = response["choices"][0]["message"]
if "function_call" in message:
    function_name = message["function_call"]["name"]
    arguments = json.loads(message["function_call"]["arguments"])
    
    # Call your function with the arguments
    function_result = get_weather(arguments["location"])
    
    # Continue the conversation with the function result
    response = client.chat_completion(
        model="ollama/mistral-small3.1:24b",
        messages=[
            {"role": "user", "content": "What's the weather in Paris?"},
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
                "content": json.dumps(function_result)
            }
        ]
    )
    
    # Final response with information from the function
    final_answer = response["choices"][0]["message"]["content"]
    print(final_answer)
```

### Function Calling Support

Different models have varying levels of support for function calling:

| Provider | Model Examples | Function Calling Support |
|----------|---------------|-------------------------|
| Ollama | mistral-small, llama3 | Yes (selected models) |
| OpenAI | gpt-4, gpt-3.5-turbo | Yes |
| Anthropic | claude-3 family | Yes |
| Cohere | command | Not yet |
| vLLM | Depends on deployed model | Depends on model |

## Working with Different Providers

Model Calling supports multiple model providers through a unified interface. Simply specify the provider in the model name.

### Provider Prefixes

| Provider | Prefix | Example |
|----------|--------|---------|
| Ollama (local) | ollama/ | "ollama/mistral-small3.1:24b" |
| vLLM (cluster) | vllm/ | "vllm/mistral-7b" |
| OpenAI | openai/ | "openai/gpt-4" |
| Anthropic | anthropic/ | "anthropic/claude-3-opus-20240229" |
| Cohere | cohere/ | "cohere/command" |

### API Keys for Hosted Providers

For hosted providers, you need to set up API keys:

1. Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
COHERE_API_KEY=your_cohere_key
```

2. The service will automatically load these keys when starting.

## Client Libraries

### Python Client

```python
from model_calling.client import SyncModelCallingClient

client = SyncModelCallingClient()

try:
    response = client.chat_completion(
        model="ollama/mistral-small3.1:24b",
        messages=[
            {"role": "user", "content": "What is quantum computing?"}
        ],
        temperature=0.7
    )
    
    print(response["choices"][0]["message"]["content"])
finally:
    client.close()
```

### Async Python Client

```python
import asyncio
from model_calling.client import ModelCallingClient

async def main():
    client = ModelCallingClient()
    
    try:
        response = await client.chat_completion(
            model="anthropic/claude-3-sonnet-20240229",
            messages=[
                {"role": "user", "content": "Explain neural networks."}
            ]
        )
        
        print(response["choices"][0]["message"]["content"])
    finally:
        await client.close()

asyncio.run(main())
```

## Monitoring & Observability

Tracing, logging, and token accounting attach through the callback interface in `model_calling.callbacks`. See the [Advanced Topics](advanced.md#monitoring-and-observability) guide for an example.

## Examples

### Basic Chat Completion

```python
response = client.chat_completion(
    model="ollama/mistral-small3.1:24b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the main challenges in AI safety?"}
    ]
)
```

### Streaming Response

```python
stream = client.chat_completion(
    model="anthropic/claude-3-sonnet-20240229",
    messages=[
        {"role": "user", "content": "Write a short story about robots."}
    ],
    stream=True
)

for chunk in stream:
    content = chunk["choices"][0]["delta"].get("content", "")
    print(content, end="", flush=True)
```

### Function Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search for records in a database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

response = client.chat_completion(
    model="openai/gpt-4",
    messages=[
        {"role": "user", "content": "Find records about renewable energy projects."}
    ],
    tools=tools
)
```

### Multi-turn Function Calling Conversation

```python
messages = [
    {"role": "user", "content": "I need to analyze the weather for my trip to Tokyo and Berlin."}
]

# First turn
response = client.chat_completion(
    model="ollama/mistral-small3.1:24b",
    messages=messages,
    tools=weather_tools
)

# Add assistant response with function call
messages.append(response["choices"][0]["message"])

# Add function result
weather_data = get_weather("Tokyo")
messages.append({
    "role": "function",
    "name": "get_weather",
    "content": json.dumps(weather_data)
})

# Next turn from the model
response = client.chat_completion(
    model="ollama/mistral-small3.1:24b",
    messages=messages,
    tools=weather_tools
)

# Continue the pattern for multiple function calls
```

### Using Different Providers

```python
# Ollama (local)
response_ollama = client.chat_completion(
    model="ollama/mistral-small3.1:24b",
    messages=[{"role": "user", "content": "Hello, who are you?"}]
)

# OpenAI
response_openai = client.chat_completion(
    model="openai/gpt-4",
    messages=[{"role": "user", "content": "Hello, who are you?"}]
)

# Anthropic
response_anthropic = client.chat_completion(
    model="anthropic/claude-3-sonnet-20240229",
    messages=[{"role": "user", "content": "Hello, who are you?"}]
)
```

---

For more detailed examples, check the [examples directory](https://github.com/rdwj/model-calling/tree/main/examples) in the repository.
