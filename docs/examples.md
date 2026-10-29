# Code Examples

Practical examples for common use cases with the model-calling library. Every example uses `model_calling.Client` directly in process. For gateway-only deployments, point the official `openai` or `anthropic` SDKs at the gateway instead (see the last section).

## Basic Chat Completion

A simple synchronous call to get a completion. The response is a typed `ChatCompletion` object with attribute access.

```python
from model_calling import Client

with Client() as client:
    response = client.completion(
        model="ollama/qwen3:4b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain neural networks in two sentences."}
        ],
        temperature=0.7
    )
    
    print(response.choices[0].message.content)
```

## Multi-Turn Conversation

Build a conversation history by appending messages. The library does not track state; you manage the history yourself.

```python
from model_calling import Client

with Client() as client:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the three laws of robotics?"}
    ]
    
    response = client.completion(model="ollama/qwen3:4b", messages=messages)
    assistant_msg = {"role": "assistant", "content": response.choices[0].message.content}
    messages.append(assistant_msg)
    print("Assistant:", assistant_msg["content"])
    
    messages.append({"role": "user", "content": "Who created these laws?"})
    response = client.completion(model="ollama/qwen3:4b", messages=messages)
    print("Assistant:", response.choices[0].message.content)
```

## Streaming Chat Completions

Stream a Chat Completions response asynchronously. Chunks arrive as `ChatCompletionChunk` objects; extract text from `delta.content`.

```python
import asyncio
from model_calling import Client

async def stream_completion():
    async with Client() as client:
        stream = await client.acompletion(
            model="ollama/qwen3:4b",
            messages=[
                {"role": "user", "content": "Write a haiku about recursion."}
            ],
            stream=True
        )
        
        full_text = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                print(text, end="", flush=True)
                full_text += text
        print("\n")
        return full_text

asyncio.run(stream_completion())
```

## Streaming Anthropic Messages

Stream an Anthropic Messages response. Events arrive as typed objects (`MessageStartEvent`, `ContentBlockDeltaEvent`, and so on). Text comes in `ContentBlockDeltaEvent` where `delta.type == "text_delta"`.

```python
import asyncio
from model_calling import Client
from model_calling.types.messages import ContentBlockDeltaEvent

async def stream_messages():
    async with Client() as client:
        stream = await client.amessages(
            model="ollama/qwen3:4b",
            messages=[
                {"role": "user", "content": "Explain photosynthesis in one paragraph."}
            ],
            max_tokens=1024,
            stream=True
        )
        
        async for event in stream:
            if isinstance(event, ContentBlockDeltaEvent) and event.delta.type == "text_delta":
                print(event.delta.text, end="", flush=True)
        print("\n")

asyncio.run(stream_messages())
```

## Function Calling with Chat Completions

A complete tool loop: the model requests a tool call via `tool_calls`, you run the tool, send the result back as a message with `role="tool"`, and get the final answer. This example uses the Chat Completions format.

```python
import json
from model_calling import Client

def get_weather(location):
    return {"location": location, "temperature": 22, "unit": "celsius", "description": "Sunny"}

with Client() as client:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
    response = client.completion(model="ollama/qwen3:4b", messages=messages, tools=tools)
    
    message = response.choices[0].message
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        result = get_weather(arguments["location"])
        
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tool_call.id, "type": "function", "function": {"name": function_name, "arguments": tool_call.function.arguments}}]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })
        
        final_response = client.completion(model="ollama/qwen3:4b", messages=messages)
        print(final_response.choices[0].message.content)
    else:
        print(message.content)
```

## Function Calling with Messages API

The same tool loop using the Anthropic Messages format. The model returns `tool_use` blocks in `content`; you send back `tool_result` blocks in a user message. Send the whole assistant content back (including thinking blocks) when building the conversation history.

```python
import json
from model_calling import Client

def get_weather(location):
    return {"location": location, "temperature": 15, "unit": "celsius", "description": "Cloudy"}

with Client() as client:
    tools = [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    ]
    
    messages = [{"role": "user", "content": "What's the weather in London?"}]
    response = client.messages(model="ollama/qwen3:4b", messages=messages, max_tokens=1024, tools=tools)
    
    tool_use_blocks = [block for block in response.content if block.type == "tool_use"]
    if tool_use_blocks:
        block = tool_use_blocks[0]
        result = get_weather(block.input["location"])
        
        # Send the typed blocks directly; the library converts them to wire dicts.
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result)
                }
            ]
        })
        
        final_response = client.messages(model="ollama/qwen3:4b", messages=messages, max_tokens=1024)
        text = next((b.text for b in final_response.content if b.type == "text"), "")
        print(text)
    else:
        text = next((b.text for b in response.content if b.type == "text"), "")
        print(text)
```

## Comparing Models Across Backends

Call different backends with the same prompt. Model routing uses `prefix/name`, with unprefixed names going to Ollama.

```python
from model_calling import Client

query = "What is a hash table?"

models = [
    "ollama/qwen3:4b",
    "vllm/gpt-oss-20b",
    "openrouter/mistralai/mistral-nemo"
]

with Client() as client:
    for model in models:
        try:
            response = client.completion(
                model=model,
                messages=[{"role": "user", "content": query}],
                temperature=0.5
            )
            print(f"{model}:")
            print(response.choices[0].message.content)
            print("-" * 50)
        except Exception as e:
            print(f"{model}: {e}")
            print("-" * 50)
```

## Small Agent Loop

An agent that calls tools until it gets a text answer. This example limits the loop to five iterations and uses the Chat Completions format.

```python
import json
from model_calling import Client

def get_time():
    from datetime import datetime
    return {"time": datetime.now().strftime("%H:%M:%S")}

def calculate(expression):
    try:
        return {"result": eval(expression, {"__builtins__": {}})}
    except Exception as e:
        return {"error": str(e)}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }
        }
    }
]

tool_map = {"get_time": get_time, "calculate": calculate}

with Client() as client:
    messages = [{"role": "user", "content": "What time is it, and what is 123 times 456?"}]
    
    for _ in range(5):
        response = client.completion(model="ollama/qwen3:4b", messages=messages, tools=tools)
        message = response.choices[0].message
        
        if not message.tool_calls:
            print(message.content)
            break
        
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in message.tool_calls
            ]
        })
        
        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            result = tool_map[fn_name](**args) if fn_name in tool_map else {"error": "unknown function"}
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)})
```

## Error Handling

The library raises typed exceptions. Check the type to handle specific errors, or catch `APIError` for all of them. `UnsupportedParamsError` names parameters the backend cannot honor; pass `drop_params=True` to send the request without them.

```python
from model_calling import Client
from model_calling.errors import (
    UnsupportedParamsError, NotFoundError, RateLimitError,
    APIConnectionError, BadRequestError
)

tools = [{"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}]
request = {
    "model": "ollama/qwen3:4b",
    "messages": [{"role": "user", "content": "What time is it?"}],
    "tools": tools,
    "tool_choice": "required",  # Ollama can't force a tool call
}

with Client() as client:
    try:
        response = client.completion(**request)
    except UnsupportedParamsError as e:
        print(f"Unsupported params: {e.params}")
        print("Retrying with drop_params=True")
        response = client.completion(**request, drop_params=True)
        print(response.choices[0].message.content)
    except NotFoundError as e:
        print(f"Model not found: {e.message} (status {e.status})")
    except RateLimitError as e:
        print(f"Rate limited: {e.message}")
    except APIConnectionError as e:
        print(f"Connection failed: {e.message} (status: {e.status})")
    except BadRequestError as e:
        print(f"Bad request: {e.message}")
```

## Batch Processing with asyncio

Process multiple requests in parallel using `asyncio.gather`. One client can serve many concurrent calls; connections pool per prefix and per event loop.

```python
import asyncio
from model_calling import Client

async def batch_completions():
    queries = [
        "What is machine learning?",
        "Explain quantum computing.",
        "What is the capital of France?",
        "How does photosynthesis work?"
    ]
    
    async with Client() as client:
        tasks = [
            client.acompletion(
                model="ollama/qwen3:4b",
                messages=[{"role": "user", "content": q}]
            )
            for q in queries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for query, result in zip(queries, results):
            if isinstance(result, Exception):
                print(f"Query: {query}\nError: {result}\n")
            else:
                print(f"Query: {query}\nResponse: {result.choices[0].message.content}\n")

asyncio.run(batch_completions())
```

## Embeddings with Cosine Similarity

Generate embeddings and search for the most similar text. The response is a typed `CreateEmbeddingResponse` with `.data[i].embedding` as a list of floats.

```python
from model_calling import Client

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)

corpus = [
    "The cat sat on the mat.",
    "A dog ran in the park.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers."
]

query = "What is AI?"

with Client() as client:
    corpus_response = client.embedding(model="ollama/nomic-embed-text", input=corpus)
    query_response = client.embedding(model="ollama/nomic-embed-text", input=query)
    
    corpus_embeddings = [item.embedding for item in corpus_response.data]
    query_embedding = query_response.data[0].embedding
    
    scores = [cosine_similarity(query_embedding, emb) for emb in corpus_embeddings]
    
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    print(f"Query: {query}")
    print(f"Best match: {corpus[best_idx]} (score: {scores[best_idx]:.3f})")
```

## Using the Gateway from Official SDKs

Point the official `openai` or `anthropic` SDK at the gateway instead of using the library directly. Install the gateway with `pip install 'model-calling[server]'`, define the `chat` alias in `model-calling.yaml` (`models: {chat: {backend: ollama, model: qwen3:4b}}`; see [Model Aliases](advanced.md#model-aliases)), and start it with `model-calling`.

```python
import openai
import anthropic

openai_client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = openai_client.chat.completions.create(
    model="chat",
    messages=[{"role": "user", "content": "Hello from the openai SDK"}]
)
print(response.choices[0].message.content)

anthropic_client = anthropic.Anthropic(base_url="http://localhost:8000", api_key="unused")
message = anthropic_client.messages.create(
    model="chat",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello from the anthropic SDK"}]
)
print(next(b.text for b in message.content if b.type == "text"))
```

The gateway serves `POST /v1/chat/completions`, `POST /v1/messages`, `POST /v1/responses` (stateless), and `POST /v1/embeddings`. It reads backend credentials from its environment (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, and so on) and does not check client keys. Requests that include `api_base`, `api_key`, or `drop_params` in the body are rejected with a 400. The gateway drops unsupported parameters by default; set `MODEL_CALLING_STRICT_PARAMS=true` in the environment to raise 400 errors instead.

## Gateway with curl

A minimal example calling the gateway directly. The response follows the Chat Completions format.

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chat",
    "messages": [{"role": "user", "content": "Hello from curl"}]
  }'
```

For streaming, add `"stream": true` and parse the SSE lines. Chat Completions streams end with `data: [DONE]`; Messages streams use `event:` and `data:` lines and end with a `message_stop` event.

## Using Callbacks

Subclass `Callback` and override the async hooks you need. Pass instances to `Client(callbacks=[...])` or per-prefix in `config`. A callback that raises is logged and ignored; it never changes the outcome of a request.

```python
import asyncio
from model_calling import Client
from model_calling.callbacks import Callback

class LoggingCallback(Callback):
    async def on_request_start(self, model, messages, **kwargs):
        print(f"Request starting for model: {model}")
    
    async def on_request_end(self, model, messages, response, **kwargs):
        if kwargs.get("cancelled"):
            print(f"Request to {model} was cancelled early")
        else:
            print(f"Request to {model} completed")
    
    async def on_error(self, model, messages, error, **kwargs):
        print(f"Error with {model}: {error}")

async def main():
    async with Client(callbacks=[LoggingCallback()]) as client:
        response = await client.acompletion(
            model="ollama/qwen3:4b",
            messages=[{"role": "user", "content": "Hello"}]
        )
        print(response.choices[0].message.content)

asyncio.run(main())
```

Every request ends with exactly one `on_request_end` or `on_error`. Streams call `on_stream_chunk` for each internal event before it is yielded. A stream stopped early (via `break` or task cancellation) gets `on_request_end(cancelled=True)`, not `on_error`.

## Overriding Configuration per Call

Override `api_base` or `api_key` on individual calls, or set them client-wide via `config`.

```python
from model_calling import Client

config = {
    "vllm": {"base_url": "http://192.168.1.100:8000/v1"},
    "openrouter": {"api_key": "sk-or-v1-custom-key"}
}

with Client(config=config) as client:
    response = client.completion(
        model="vllm/gpt-oss-20b",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(response.choices[0].message.content)
    
    response = client.completion(
        model="openrouter/mistralai/mistral-nemo",
        messages=[{"role": "user", "content": "Hello"}],
        api_key="sk-or-v1-override"
    )
    print(response.choices[0].message.content)
```

The `config` parameter maps a prefix to settings merged over the defaults: `base_url`, `api_key`, `headers`, `callbacks`, and `transport` (an httpx transport, mostly for tests). Per-call `api_base` and `api_key` override both the client config and environment variables.
