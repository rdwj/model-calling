# Quickstart Guide

This guide helps you get started with Model Calling quickly.

## Installation

### 1. Install the Package

```bash
pip install model-calling
```

### 2. Set Up API Keys (Optional)

For hosted providers, create a `.env` file with your API keys:

```bash
# Copy the example .env file
cp .env.example .env

# Edit the file with your API keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
COHERE_API_KEY=your_cohere_key
```

### 3. Start the Service

```bash
# Start the service on http://localhost:8000
python -m model_calling
```

## Using the API

### Basic Example

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "ollama/mistral-small3.1:24b",  # Use any model from any provider
        "messages": [
            {"role": "user", "content": "What is machine learning?"}
        ]
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### Using the Client Library

```python
from model_calling.client import SyncModelCallingClient

client = SyncModelCallingClient()

try:
    response = client.chat_completion(
        model="ollama/mistral-small3.1:24b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning?"}
        ]
    )
    
    print(response["choices"][0]["message"]["content"])
finally:
    client.close()
```

### Function Calling

```python
from model_calling.client import SyncModelCallingClient
import json

client = SyncModelCallingClient()

# Define tools
tools = [
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

try:
    # Send request
    response = client.chat_completion(
        model="ollama/mistral-small3.1:24b",
        messages=[
            {"role": "user", "content": "What's the weather in Paris?"}
        ],
        tools=tools
    )
    
    # Process function call
    message = response["choices"][0]["message"]
    if "function_call" in message:
        function_name = message["function_call"]["name"]
        arguments = json.loads(message["function_call"]["arguments"])
        
        print(f"Function called: {function_name}")
        print(f"Arguments: {arguments}")
        
        # In a real app, you would call your actual function here
        # weather_data = get_weather(arguments["location"])
        
        # For this example, we'll simulate a response
        weather_data = {
            "temperature": 22,
            "unit": "celsius",
            "description": "Partly cloudy"
        }
        
        # Continue the conversation with the function result
        final_response = client.chat_completion(
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
                    "content": json.dumps(weather_data)
                }
            ]
        )
        
        print("\nFinal response:")
        print(final_response["choices"][0]["message"]["content"])
    else:
        print(message["content"])
finally:
    client.close()
```

### Using Different Providers

```python
from model_calling.client import SyncModelCallingClient

client = SyncModelCallingClient()

try:
    # Ollama (local)
    response = client.chat_completion(
        model="ollama/mistral-small3.1:24b",
        messages=[{"role": "user", "content": "What is quantum computing?"}]
    )
    
    print("Ollama response:")
    print(response["choices"][0]["message"]["content"])
    
    # OpenAI (if API key is set)
    response = client.chat_completion(
        model="openai/gpt-4",
        messages=[{"role": "user", "content": "What is quantum computing?"}]
    )
    
    print("\nOpenAI response:")
    print(response["choices"][0]["message"]["content"])
    
    # Anthropic (if API key is set)
    response = client.chat_completion(
        model="anthropic/claude-3-sonnet-20240229",
        messages=[{"role": "user", "content": "What is quantum computing?"}]
    )
    
    print("\nAnthropic response:")
    print(response["choices"][0]["message"]["content"])
finally:
    client.close()
```

### Streaming Responses

```python
import asyncio
from model_calling.client import ModelCallingClient

async def stream_example():
    client = ModelCallingClient()
    
    try:
        stream = await client.chat_completion(
            model="ollama/mistral-small3.1:24b",
            messages=[
                {"role": "user", "content": "Write a short story about robots."}
            ],
            stream=True
        )
        
        print("Streaming response:")
        async for chunk in stream:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta and delta["content"]:
                    print(delta["content"], end="", flush=True)
        print("\n")
    finally:
        await client.close()

# Run the async function
asyncio.run(stream_example())
```

## Next Steps

- Check out the [API Reference](api-reference.md) for detailed documentation
- Explore more [Examples](examples.md)
- Learn about [Advanced Topics](advanced.md)
