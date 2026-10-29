"""
Example of using hosted providers (OpenAI, Anthropic, Cohere).
"""
import asyncio
import json
import os
from dotenv import load_dotenv
from model_calling.client import ModelCallingClient

# Load environment variables
load_dotenv()


async def test_hosted_providers():
    """Test different hosted providers"""
    client = ModelCallingClient()
    
    try:
        # 1. Test OpenAI
        if os.environ.get("OPENAI_API_KEY"):
            print("\n=== Testing OpenAI (GPT-4) ===")
            response = await client.chat_completion(
                model="openai/gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What are three benefits of quantum computing?"}
                ],
                temperature=0.7
            )
            
            print("Response:")
            print(response["choices"][0]["message"]["content"])
            
            # Test function calling with OpenAI
            print("\n=== Testing OpenAI with Function Calling ===")
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
                model="openai/gpt-4-turbo",
                messages=[
                    {"role": "user", "content": "What's the weather like in Paris?"}
                ],
                tools=tools,
                temperature=0.7
            )
            
            message = response["choices"][0]["message"]
            if "function_call" in message:
                print(f"Function call requested: {message['function_call']['name']}")
                print(f"Arguments: {message['function_call']['arguments']}")
        else:
            print("Skipping OpenAI test - no API key found")
        
        # 2. Test Anthropic
        if os.environ.get("ANTHROPIC_API_KEY"):
            print("\n=== Testing Anthropic (Claude) ===")
            response = await client.chat_completion(
                model="anthropic/claude-haiku-5-5",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Explain the concept of reinforcement learning in simple terms."}
                ],
                temperature=0.7
            )
            
            print("Response:")
            print(response["choices"][0]["message"]["content"])
            
            # Test function calling with Anthropic
            print("\n=== Testing Anthropic with Function Calling ===")
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
                model="anthropic/claude-haiku-5-5",
                messages=[
                    {"role": "user", "content": "What's the weather like in Tokyo?"}
                ],
                tools=tools,
                temperature=0.7
            )
            
            message = response["choices"][0]["message"]
            if "function_call" in message:
                print(f"Function call requested: {message['function_call']['name']}")
                print(f"Arguments: {message['function_call']['arguments']}")
        else:
            print("Skipping Anthropic test - no API key found")
        
        # 3. Test Cohere
        if os.environ.get("COHERE_API_KEY"):
            print("\n=== Testing Cohere ===")
            response = await client.chat_completion(
                model="cohere/command",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What are three interesting facts about coral reefs?"}
                ],
                temperature=0.7
            )
            
            print("Response:")
            print(response["choices"][0]["message"]["content"])
        else:
            print("Skipping Cohere test - no API key found")
            
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_hosted_providers())
