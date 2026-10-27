# Code Examples

This page provides practical code examples for common use cases with the Model Calling API.

## Basic Chat Completion

Simple example of generating a text response from a model.

```python
from model_calling.client import SyncModelCallingClient

client = SyncModelCallingClient()

try:
    response = client.chat_completion(
        model="ollama/mistral-small3.1:24b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain the concept of neural networks in simple terms."}
        ],
        temperature=0.7
    )
    
    print(response["choices"][0]["message"]["content"])
finally:
    client.close()
```

## Multi-turn Conversation

Example of a multi-turn conversation with the model.

```python
from model_calling.client import SyncModelCallingClient

client = SyncModelCallingClient()

try:
    # Conversation history
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the three laws of robotics?"}
    ]
    
    # First user turn
    response = client.chat_completion(
        model="ollama/mistral-small3.1:24b",
        messages=messages
    )
    
    assistant_message = response["choices"][0]["message"]
    messages.append(assistant_message)
    print("Assistant:", assistant_message["content"])
    
    # Second user turn
    messages.append({"role": "user", "content": "Who created these laws?"})
    
    response = client.chat_completion(
        model="ollama/mistral-small3.1:24b",
        messages=messages
    )
    
    assistant_message = response["choices"][0]["message"]
    messages.append(assistant_message)
    print("Assistant:", assistant_message["content"])
    
    # Third user turn
    messages.append({"role": "user", "content": "What year were they published?"})
    
    response = client.chat_completion(
        model="ollama/mistral-small3.1:24b",
        messages=messages
    )
    
    assistant_message = response["choices"][0]["message"]
    print("Assistant:", assistant_message["content"])
finally:
    client.close()
```

## Streaming Response

Example of streaming a response from the model.

```python
import asyncio
from model_calling.client import ModelCallingClient

async def stream_example():
    client = ModelCallingClient()
    
    try:
        stream = await client.chat_completion(
            model="ollama/mistral-small3.1:24b",
            messages=[
                {"role": "system", "content": "You are a creative assistant."},
                {"role": "user", "content": "Write a short poem about artificial intelligence."}
            ],
            stream=True
        )
        
        # Process streaming chunks
        full_response = ""
        print("Streaming response:")
        async for chunk in stream:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta and delta["content"]:
                    content = delta["content"]
                    print(content, end="", flush=True)
                    full_response += content
        
        print("\n\nFull response:")
        print(full_response)
    finally:
        await client.close()

# Run the async function
asyncio.run(stream_example())
```

## Function Calling

Example of using function calling to get external data.

```python
from model_calling.client import SyncModelCallingClient
import json

# Define a function to get weather data (in a real app, this would call a weather API)
def get_weather(location):
    print(f"Getting weather for {location}...")
    # Simulated weather data
    return {
        "location": location,
        "temperature": 22,
        "unit": "celsius",
        "description": "Partly cloudy",
        "humidity": 65
    }

client = SyncModelCallingClient()

# Define tool schema
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
    # Initial request that may trigger function calling
    response = client.chat_completion(
        model="ollama/mistral-small3.1:24b",
        messages=[
            {"role": "user", "content": "What's the weather like in Tokyo today?"}
        ],
        tools=tools
    )
    
    # Check if the model wants to call a function
    message = response["choices"][0]["message"]
    if "function_call" in message:
        # Extract function call details
        function_name = message["function_call"]["name"]
        arguments = json.loads(message["function_call"]["arguments"])
        
        print(f"Function call requested: {function_name}")
        print(f"Arguments: {arguments}")
        
        # Execute the function
        if function_name == "get_weather":
            function_result = get_weather(arguments["location"])
        else:
            function_result = {"error": f"Unknown function: {function_name}"}
        
        # Continue the conversation with the function result
        final_response = client.chat_completion(
            model="ollama/mistral-small3.1:24b",
            messages=[
                {"role": "user", "content": "What's the weather like in Tokyo today?"},
                {
                    "role": "assistant", 
                    "content": "", 
                    "function_call": {
                        "name": function_name,
                        "arguments": json.dumps(arguments)
                    }
                },
                {
                    "role": "function", 
                    "name": function_name, 
                    "content": json.dumps(function_result)
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

## Multiple Function Calls

Example of handling multiple function calls in a conversation.

```python
from model_calling.client import SyncModelCallingClient
import json

# Define our tool functions
def get_current_weather(location):
    # Simulate weather API call
    return {
        "location": location,
        "temperature": 22,
        "unit": "celsius",
        "description": "Partly cloudy"
    }

def get_population(city):
    # Simulate population API call
    populations = {
        "Tokyo": 37.4,
        "New York": 8.8,
        "London": 9.0,
        "Paris": 2.2,
        "Sydney": 5.3
    }
    return {
        "city": city,
        "population": populations.get(city, "unknown"),
        "unit": "million"
    }

# Tool definitions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_population",
            "description": "Get the population of a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

client = SyncModelCallingClient()

try:
    # Start conversation
    messages = [
        {"role": "user", "content": "I'm planning a trip to Tokyo. Can you tell me about the weather and population there?"}
    ]
    
    # Process multiple turns with function calls
    max_turns = 3
    current_turn = 0
    
    while current_turn < max_turns:
        response = client.chat_completion(
            model="ollama/mistral-small3.1:24b",
            messages=messages,
            tools=tools
        )
        
        assistant_message = response["choices"][0]["message"]
        messages.append(assistant_message)
        
        # Check if function call was requested
        if "function_call" in assistant_message:
            function_name = assistant_message["function_call"]["name"]
            arguments = json.loads(assistant_message["function_call"]["arguments"])
            
            print(f"\nFunction call: {function_name}")
            print(f"Arguments: {arguments}")
            
            # Call the appropriate function
            if function_name == "get_current_weather":
                result = get_current_weather(arguments["location"])
            elif function_name == "get_population":
                result = get_population(arguments["city"])
            else:
                result = {"error": f"Unknown function: {function_name}"}
            
            # Add function result to messages
            function_message = {
                "role": "function",
                "name": function_name,
                "content": json.dumps(result)
            }
            messages.append(function_message)
            print(f"Function result: {result}")
        else:
            # No function call, just a regular message
            print(f"\nAssistant: {assistant_message['content']}")
            break
        
        current_turn += 1
    
    # If we used up all turns but still haven't gotten a final answer
    if current_turn == max_turns:
        # Get final synthesis with all the function results
        final_response = client.chat_completion(
            model="ollama/mistral-small3.1:24b",
            messages=messages
        )
        
        print(f"\nFinal answer: {final_response['choices'][0]['message']['content']}")
finally:
    client.close()
```

## Comparing Models

Example of comparing responses from different models.

```python
from model_calling.client import SyncModelCallingClient

client = SyncModelCallingClient()

# Function to get response from a model
def get_model_response(model, query):
    response = client.chat_completion(
        model=model,
        messages=[
            {"role": "user", "content": query}
        ]
    )
    return response["choices"][0]["message"]["content"]

try:
    # The query to compare
    query = "Explain how transformers work in deep learning in 3-4 sentences."
    
    # Compare models
    models = [
        "ollama/mistral-small3.1:24b",
        "ollama/llama3:8b",
        "openai/gpt-4-turbo",
        "anthropic/claude-3-sonnet-20240229"
    ]
    
    print(f"Query: {query}\n")
    
    for model in models:
        try:
            response = get_model_response(model, query)
            print(f"{model}:\n{response}\n{'-'*50}\n")
        except Exception as e:
            print(f"{model}: Error - {str(e)}\n{'-'*50}\n")
finally:
    client.close()
```

## Using REST API Directly

Example of using the REST API without the client library.

```python
import requests
import json

# Function calling example with direct REST API
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

# Initial request
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "ollama/mistral-small3.1:24b",
        "messages": [
            {"role": "user", "content": "What's the weather in San Francisco?"}
        ],
        "tools": tools
    }
)

data = response.json()
message = data["choices"][0]["message"]

# Check for function call
if "function_call" in message:
    function_name = message["function_call"]["name"]
    arguments = json.loads(message["function_call"]["arguments"])
    
    print(f"Function call: {function_name}")
    print(f"Arguments: {arguments}")
    
    # Simulate function call
    weather_data = {
        "temperature": 15,
        "unit": "celsius",
        "description": "Foggy and cool"
    }
    
    # Continue conversation with function result
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "ollama/mistral-small3.1:24b",
            "messages": [
                {"role": "user", "content": "What's the weather in San Francisco?"},
                {
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": function_name,
                        "arguments": json.dumps(arguments)
                    }
                },
                {
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps(weather_data)
                }
            ]
        }
    )
    
    final_data = response.json()
    print("\nFinal response:")
    print(final_data["choices"][0]["message"]["content"])
else:
    print(message["content"])
```

## Building an Agent

Example of building a simple agent that can use multiple functions.

```python
from model_calling.client import SyncModelCallingClient
import json
import datetime

class SimpleAgent:
    def __init__(self):
        self.client = SyncModelCallingClient()
        self.model = "ollama/mistral-small3.1:24b"
        self.messages = []
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current time",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
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
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform a calculation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Mathematical expression to calculate"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]
        
        # Tool implementation map
        self.tool_map = {
            "get_current_time": self._get_current_time,
            "get_weather": self._get_weather,
            "calculate": self._calculate
        }
    
    def _get_current_time(self, args=None):
        """Get the current time"""
        now = datetime.datetime.now()
        return {
            "time": now.strftime("%H:%M:%S"),
            "date": now.strftime("%Y-%m-%d")
        }
    
    def _get_weather(self, args):
        """Simulate getting weather data"""
        location = args.get("location", "unknown")
        return {
            "location": location,
            "temperature": 22,
            "unit": "celsius",
            "description": "Partly cloudy"
        }
    
    def _calculate(self, args):
        """Perform a calculation"""
        expression = args.get("expression", "")
        try:
            result = eval(expression, {"__builtins__": {}})
            return {"result": result}
        except Exception as e:
            return {"error": f"Error calculating: {str(e)}"}
    
    def chat(self, user_message):
        """Process a user message and generate a response"""
        # Add user message to history
        self.messages.append({"role": "user", "content": user_message})
        
        # Get response from model
        response = self.client.chat_completion(
            model=self.model,
            messages=self.messages,
            tools=self.tools
        )
        
        assistant_message = response["choices"][0]["message"]
        self.messages.append(assistant_message)
        
        # Check if function call was requested
        if "function_call" in assistant_message:
            # Process function call
            return self._process_function_call()
        else:
            # Return text response
            return assistant_message["content"]
    
    def _process_function_call(self):
        """Process function calls, possibly multiple in a row"""
        max_function_calls = 5
        num_calls = 0
        
        while num_calls < max_function_calls:
            # Get the latest assistant message
            assistant_message = self.messages[-1]
            
            # Check if it's a function call
            if "function_call" not in assistant_message:
                # Got a regular message, return it
                return assistant_message["content"]
            
            # Extract function details
            function_name = assistant_message["function_call"]["name"]
            arguments_str = assistant_message["function_call"]["arguments"]
            arguments = json.loads(arguments_str)
            
            # Call the function
            if function_name in self.tool_map:
                function_result = self.tool_map[function_name](arguments)
            else:
                function_result = {"error": f"Unknown function: {function_name}"}
            
            # Add function result to messages
            self.messages.append({
                "role": "function",
                "name": function_name,
                "content": json.dumps(function_result)
            })
            
            # Get next response from model
            response = self.client.chat_completion(
                model=self.model,
                messages=self.messages,
                tools=self.tools
            )
            
            assistant_message = response["choices"][0]["message"]
            self.messages.append(assistant_message)
            
            num_calls += 1
        
        # If we hit the limit, return whatever we have
        return assistant_message["content"]
    
    def close(self):
        """Clean up resources"""
        self.client.close()

# Example usage
agent = SimpleAgent()

try:
    print("Agent: Hello! How can I help you today?")
    
    # Example queries
    queries = [
        "What time is it?",
        "What's the weather in Tokyo?",
        "What's 123 * 456?",
        "If it's 22°C in Tokyo, what would that be in Fahrenheit?"
    ]
    
    for query in queries:
        print(f"\nUser: {query}")
        response = agent.chat(query)
        print(f"Agent: {response}")
finally:
    agent.close()
```

## Choosing Between Providers

Example of how to choose the best provider for a specific task:

```python
from model_calling.client import SyncModelCallingClient

# Provider selection logic
def select_provider(task_type, content_length=None, importance=None):
    """Select the best provider/model for a specific task"""
    
    if task_type == "creative":
        if importance == "high":
            return "anthropic/claude-3-opus-20240229"
        else:
            return "ollama/llama3:8b"
    
    elif task_type == "coding":
        if content_length == "long":
            return "openai/gpt-4-turbo"
        else:
            return "ollama/codellama:7b"
    
    elif task_type == "factual":
        return "anthropic/claude-3-sonnet-20240229"
    
    else:  # Default
        return "ollama/mistral-small3.1:24b"

client = SyncModelCallingClient()

try:
    # Example tasks
    tasks = [
        {"type": "creative", "query": "Write a short poem about AI", "importance": "low"},
        {"type": "coding", "query": "Write a Python function to calculate Fibonacci numbers", "content_length": "short"},
        {"type": "factual", "query": "Explain how quantum computing works"}
    ]
    
    for task in tasks:
        # Select the appropriate provider
        provider_model = select_provider(
            task["type"], 
            task.get("content_length"), 
            task.get("importance")
        )
        
        print(f"\nTask: {task['query']}")
        print(f"Selected model: {provider_model}")
        
        # Call the model
        response = client.chat_completion(
            model=provider_model,
            messages=[
                {"role": "user", "content": task["query"]}
            ]
        )
        
        print("Response:")
        print(response["choices"][0]["message"]["content"])
        print("-" * 50)
finally:
    client.close()
```

## Error Handling

Example of robust error handling:

```python
from model_calling.client import SyncModelCallingClient

client = SyncModelCallingClient()

def run_with_fallbacks(query, models):
    """Try to get a response from multiple models with fallbacks"""
    last_error = None
    
    for model in models:
        try:
            response = client.chat_completion(
                model=model,
                messages=[
                    {"role": "user", "content": query}
                ],
                timeout=10.0  # Set a timeout for the request
            )
            
            # If we get here, the request was successful
            return {
                "success": True,
                "model": model,
                "content": response["choices"][0]["message"]["content"]
            }
        
        except Exception as e:
            print(f"Error with model {model}: {str(e)}")
            last_error = str(e)
            # Continue to the next model
    
    # If we get here, all models failed
    return {
        "success": False,
        "error": last_error
    }

try:
    # Query to run
    query = "What are three interesting scientific discoveries in the last decade?"
    
    # Model fallback chain
    fallback_chain = [
        "ollama/mistral-small3.1:24b",  # First choice: local model
        "openai/gpt-3.5-turbo",         # Fallback 1: cheaper API model 
        "anthropic/claude-3-haiku-20240307"  # Fallback 2: another provider
    ]
    
    result = run_with_fallbacks(query, fallback_chain)
    
    if result["success"]:
        print(f"Successful response from {result['model']}:")
        print(result["content"])
    else:
        print(f"All models failed. Last error: {result['error']}")
finally:
    client.close()
```

## Batch Processing

Example of processing multiple queries in parallel:

```python
import asyncio
from model_calling.client import ModelCallingClient

async def process_queries(queries, model):
    """Process multiple queries in parallel"""
    client = ModelCallingClient()
    
    try:
        # Create tasks for all queries
        tasks = []
        for query in queries:
            task = asyncio.create_task(client.chat_completion(
                model=model,
                messages=[
                    {"role": "user", "content": query}
                ]
            ))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle error
                processed_results.append({
                    "query": queries[i],
                    "success": False,
                    "error": str(result)
                })
            else:
                # Process successful response
                processed_results.append({
                    "query": queries[i],
                    "success": True,
                    "content": result["choices"][0]["message"]["content"]
                })
        
        return processed_results
    
    finally:
        await client.close()

# Example usage
async def main():
    # List of queries to process
    queries = [
        "What is machine learning?",
        "Explain quantum computing",
        "What is the capital of France?",
        "How does photosynthesis work?",
        "What is the Pythagorean theorem?"
    ]
    
    # Process queries in parallel
    results = await process_queries(queries, "ollama/mistral-small3.1:24b")
    
    # Display results
    for i, result in enumerate(results):
        print(f"\nQuery {i+1}: {result['query']}")
        if result['success']:
            print(f"Response: {result['content'][:100]}...")  # First 100 chars
        else:
            print(f"Error: {result['error']}")

# Run the async function
asyncio.run(main())
```

## More Examples

For more examples, see the [examples directory](https://github.com/rdwj/model-calling/tree/main/examples) in the repository.
