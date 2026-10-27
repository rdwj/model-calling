#!/bin/bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/mistral-large:latest",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant that checks the weather."},
      {"role": "user", "content": "What is the weather like in Paris, France?"}
    ],
    "tools": [
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
                "description": "Boston, MA"
              }
            },
            "required": ["location"]
          }
        }
      }
    ],
    "temperature": 0.7,
    "stream": false
  }'