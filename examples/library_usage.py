"""
Example of using model-calling as an imported package in your application.
"""

import json
from model_calling import Client

def main():
    """Simple example of using model-calling as a library"""
    # Create a synchronous client
    with Client() as client:
        # Call the model
        response = client.completion(
            model="ollama/qwen3:0.6b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain the concept of APIs in simple terms."}
            ],
            temperature=0.7
        )

        # Extract and print the response
        content = response.choices[0].message.content
        print(f"Model response: {content}")

        # Try function calling
        print("\nTesting function calling...")

        weather_tool = [
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

        # Call with a function
        response = client.completion(
            model="ollama/qwen3:0.6b",
            messages=[
                {"role": "user", "content": "What's the weather in Boston?"}
            ],
            tools=weather_tool
        )

        # Process function call
        message = response.choices[0].message
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            print(f"Tool called: {tool_call.function.name}")
            print(f"Arguments: {tool_call.function.arguments}")

            # Simulate function execution
            weather_result = {
                "temperature": 15,
                "unit": "celsius",
                "description": "Cloudy with occasional showers"
            }

            # Send function result back
            final_response = client.completion(
                model="ollama/qwen3:0.6b",
                messages=[
                    {"role": "user", "content": "What's the weather in Boston?"},
                    {
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": json.dumps({"location": "Boston, MA"})
                                }
                            }
                        ]
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(weather_result)
                    }
                ]
            )

            print(f"\nFinal response: {final_response.choices[0].message.content}")

if __name__ == "__main__":
    main()
