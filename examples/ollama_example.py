"""
Sample usage of model-calling service with Ollama.
"""
import asyncio
import json
from model_calling.client import ModelCallingClient


async def simple_chat_example():
    """Simple chat example with Ollama model"""
    client = ModelCallingClient()
    
    try:
        # Basic chat completion
        response = await client.chat_completion(
            model="ollama/mistral-small3.1:24b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ]
        )
        
        print("Basic chat response:")
        print(response["choices"][0]["message"]["content"])
        print("\n" + "-" * 50 + "\n")
        
        # Function calling example
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
        
        response = await client.chat_completion(
            model="ollama/mistral-small3.1:24b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the weather like in Paris?"}
            ],
            tools=tools
        )
        
        print("Function calling response:")
        
        # Check if we got a function call or regular response
        message = response["choices"][0]["message"]
        if "function_call" in message:
            function_call = message["function_call"]
            print(f"Function: {function_call['name']}")
            print(f"Arguments: {function_call['arguments']}")
            
            # In a real app, you would call the actual function here
            # For this example, we'll simulate the function response
            function_response = {"temperature": 22, "unit": "celsius", "description": "Partly cloudy"}
            
            # Continue the conversation with the function result
            response = await client.chat_completion(
                model="ollama/mistral-small3.1:24b",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What's the weather like in Paris?"},
                    {"role": "assistant", "content": "", "function_call": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "Paris, France"})
                    }},
                    {"role": "function", "name": "get_weather", "content": json.dumps(function_response)}
                ]
            )
            
            print("\nFinal response with function result:")
            print(response["choices"][0]["message"]["content"])
        else:
            print(message["content"])
    
    finally:
        await client.close()


async def streaming_example():
    """Example of streaming responses"""
    client = ModelCallingClient()
    
    try:
        print("Streaming response:")
        
        # Set up streaming request
        stream = await client.chat_completion(
            model="ollama/mistral-small3.1:24b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a short poem about programming."}
            ],
            stream=True
        )
        
        # Process streaming chunks
        full_response = ""
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


async def main():
    """Run all examples"""
    print("=== Simple Chat Example ===")
    await simple_chat_example()
    
    print("\n\n=== Streaming Example ===")
    await streaming_example()


if __name__ == "__main__":
    asyncio.run(main())
