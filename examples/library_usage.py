"""
Example of using model-calling as an imported package in your application.
"""

# Import as a library
from model_calling.client import SyncModelCallingClient
import json

def main():
    """Simple example of using model-calling as a library"""
    # Create a synchronous client
    client = SyncModelCallingClient()
    
    try:
        # Call the model
        response = client.chat_completion(
            model="ollama/mistral-small3.1:24b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain the concept of APIs in simple terms."}
            ],
            temperature=0.7
        )
        
        # Extract and print the response
        content = response["choices"][0]["message"]["content"]
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
        response = client.chat_completion(
            model="ollama/mistral-small3.1:24b",
            messages=[
                {"role": "user", "content": "What's the weather in Boston?"}
            ],
            tools=weather_tool
        )
        
        # Process function call
        message = response["choices"][0]["message"]
        if "function_call" in message:
            function_call = message["function_call"]
            print(f"Function called: {function_call['name']}")
            print(f"Arguments: {function_call['arguments']}")
            
            # Simulate function execution
            weather_result = {
                "temperature": 15,
                "unit": "celsius",
                "description": "Cloudy with occasional showers"
            }
            
            # Send function result back
            final_response = client.chat_completion(
                model="ollama/mistral-small3.1:24b",
                messages=[
                    {"role": "user", "content": "What's the weather in Boston?"},
                    {
                        "role": "assistant", 
                        "content": "", 
                        "function_call": {
                            "name": "get_weather",
                            "arguments": json.dumps({"location": "Boston, MA"})
                        }
                    },
                    {
                        "role": "function",
                        "name": "get_weather",
                        "content": json.dumps(weather_result)
                    }
                ]
            )
            
            print(f"\nFinal response: {final_response['choices'][0]['message']['content']}")
        
    finally:
        client.close()

if __name__ == "__main__":
    main()
