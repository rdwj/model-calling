"""
Example of integrating model-calling as a library in your own applications.
"""
import asyncio
import json
from typing import Dict, Any, List, Callable, Awaitable, Optional

# Import the client
from model_calling.client import ModelCallingClient


class WeatherAgent:
    """Example agent that can look up weather information"""
    
    def __init__(self, model: str = "ollama/mistral-small3.1:24b"):
        """Initialize with the model to use"""
        self.model = model
        self.client = ModelCallingClient()
        self.conversation_history = []
        
        # Register available tools
        self.tools = [
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
        
        # Map of tool names to functions
        self.tool_functions = {
            "get_weather": self.get_weather
        }
    
    async def get_weather(self, location: str) -> Dict[str, Any]:
        """Get weather for a location (simulated)"""
        print(f"Getting weather for {location}...")
        # In a real app, this would call a weather API
        # For this example, we'll simulate the response
        
        weather_data = {
            "temperature": 22,
            "unit": "celsius",
            "description": "Partly cloudy with a chance of rain",
            "humidity": 65
        }
        
        return weather_data
    
    async def chat(self, message: str, system_prompt: Optional[str] = None) -> str:
        """Process a user message and return a response"""
        try:
            # Setup messages
            if not self.conversation_history:
                # Start a new conversation
                if system_prompt:
                    self.conversation_history.append({"role": "system", "content": system_prompt})
            
            # Add user message
            self.conversation_history.append({"role": "user", "content": message})
            
            # Get response from model
            response = await self.client.chat_completion(
                model=self.model,
                messages=self.conversation_history,
                tools=self.tools
            )
            
            # Process the response
            assistant_message = response["choices"][0]["message"]
            
            # Check if the model wants to call a function
            if "function_call" in assistant_message:
                function_call = assistant_message["function_call"]
                function_name = function_call["name"]
                
                # Parse arguments
                arguments = json.loads(function_call["arguments"])
                
                # Add assistant's function call to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message.get("content", ""),
                    "function_call": function_call
                })
                
                # Call the appropriate function
                if function_name in self.tool_functions:
                    function_result = await self.tool_functions[function_name](**arguments)
                    
                    # Add function result to history
                    self.conversation_history.append({
                        "role": "function",
                        "name": function_name,
                        "content": json.dumps(function_result)
                    })
                    
                    # Get final response with function result
                    final_response = await self.client.chat_completion(
                        model=self.model,
                        messages=self.conversation_history
                    )
                    
                    final_message = final_response["choices"][0]["message"]
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": final_message["content"]
                    })
                    
                    return final_message["content"]
                else:
                    return f"Error: Function {function_name} not found"
            else:
                # Standard response (no function call)
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message["content"]
                })
                
                return assistant_message["content"]
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def close(self):
        """Close the client"""
        await self.client.close()


async def main():
    """Run a simple conversation with the weather agent"""
    agent = WeatherAgent()
    
    try:
        system_prompt = "You are a helpful weather assistant. Always provide accurate and concise weather information."
        
        # First user message
        response = await agent.chat("Hi, I'd like to know the weather in San Francisco.", system_prompt)
        print(f"Agent: {response}")
        
        # Follow-up question
        response = await agent.chat("What about the weather in New York?")
        print(f"Agent: {response}")
        
        # Non-weather question
        response = await agent.chat("Can you also tell me about the history of San Francisco?")
        print(f"Agent: {response}")
    
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
