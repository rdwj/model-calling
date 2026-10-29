"""
Example of using hosted providers (OpenAI, Anthropic).
"""
import asyncio
import os
from dotenv import load_dotenv
from model_calling import Client

# Load environment variables
load_dotenv()


async def test_hosted_providers():
    """Test different hosted providers"""
    async with Client() as client:
        # 1. Test OpenAI
        if os.environ.get("OPENAI_API_KEY"):
            print("\n=== Testing OpenAI (GPT-4) ===")
            response = await client.acompletion(
                model="openai/gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What are three benefits of quantum computing?"}
                ],
                temperature=0.7
            )

            print("Response:")
            print(response.choices[0].message.content)

            # Test function calling with OpenAI
            print("\n=== Testing OpenAI with Tool Calling ===")
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

            response = await client.acompletion(
                model="openai/gpt-4-turbo",
                messages=[
                    {"role": "user", "content": "What's the weather like in Paris?"}
                ],
                tools=tools,
                temperature=0.7
            )

            message = response.choices[0].message
            if message.tool_calls:
                print(f"Tool called: {message.tool_calls[0].function.name}")
                print(f"Arguments: {message.tool_calls[0].function.arguments}")
        else:
            print("Skipping OpenAI test - no API key found")
        
        # 2. Test Anthropic
        if os.environ.get("ANTHROPIC_API_KEY"):
            print("\n=== Testing Anthropic (Claude) ===")
            response = await client.acompletion(
                model="anthropic/claude-haiku-5-5",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Explain the concept of reinforcement learning in simple terms."}
                ],
                temperature=0.7
            )

            print("Response:")
            print(response.choices[0].message.content)

            # Test tool calling with Anthropic
            print("\n=== Testing Anthropic with Tool Calling ===")
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

            response = await client.acompletion(
                model="anthropic/claude-haiku-5-5",
                messages=[
                    {"role": "user", "content": "What's the weather like in Tokyo?"}
                ],
                tools=tools,
                temperature=0.7
            )

            message = response.choices[0].message
            if message.tool_calls:
                print(f"Tool called: {message.tool_calls[0].function.name}")
                print(f"Arguments: {message.tool_calls[0].function.arguments}")
        else:
            print("Skipping Anthropic test - no API key found")

        # Cohere backend is not available in the library


if __name__ == "__main__":
    asyncio.run(test_hosted_providers())
