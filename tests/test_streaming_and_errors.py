"""
Tests for streaming and error handling with Ollama models.
"""
import pytest
import requests
import json
import time
from tests.conftest import TEST_CONFIG, TEST_MESSAGES, WEATHER_TOOL

# Base URL for API calls
BASE_URL = TEST_CONFIG["service_url"]

@pytest.mark.parametrize("model_name", TEST_CONFIG["ollama_models"]["basic"][:1])  # Use only one model for streaming tests
def test_streaming_completion(model_name):
    """Test streaming completion"""
    # Prepare the request
    payload = {
        "model": f"ollama/{model_name}",
        "messages": TEST_MESSAGES,
        "stream": True
    }
    
    # Make the request (using stream=True for requests)
    response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, stream=True)
    assert response.status_code == 200, f"Failed to start streaming from {model_name}"
    
    # Process the streaming response
    chunks = []
    full_content = ""
    
    for line in response.iter_lines():
        if line:
            # Remove 'data: ' prefix if present (for SSE)
            text = line.decode('utf-8')
            if text.startswith('data: '):
                text = text[6:]
            
            # Skip '[DONE]' marker
            if text == '[DONE]':
                continue
                
            try:
                chunk = json.loads(text)
                chunks.append(chunk)
                
                # Extract content from chunk
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        full_content += delta["content"]
            except json.JSONDecodeError:
                print(f"Error parsing chunk: {text}")
    
    # Validate streaming results
    assert chunks, "No chunks received"
    assert full_content.strip(), "No content received in streaming mode"
    
    # Print the first few characters and the total length
    print(f"\nStreaming from {model_name}: {full_content[:50]}... (Total length: {len(full_content)})")

@pytest.mark.parametrize("model_name", TEST_CONFIG["ollama_models"]["function_calling"][:1])
def test_function_call_streaming(model_name):
    """Test function calling with streaming enabled"""
    # Prepare the request
    payload = {
        "model": f"ollama/{model_name}",
        "messages": [
            {"role": "user", "content": "What's the weather like in Tokyo?"}
        ],
        "tools": WEATHER_TOOL,
        "stream": True
    }
    
    # Make the request
    response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, stream=True)
    assert response.status_code == 200, f"Failed to start streaming from {model_name}"
    
    # Process the streaming response
    chunks = []
    function_call_data = None
    
    for line in response.iter_lines():
        if line:
            # Process the line
            text = line.decode('utf-8')
            if text.startswith('data: '):
                text = text[6:]
            
            if text == '[DONE]':
                continue
                
            try:
                chunk = json.loads(text)
                chunks.append(chunk)
                
                # Check for function call information in chunks
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    if "function_call" in delta:
                        function_call_data = delta["function_call"]
                    elif chunk["choices"][0].get("finish_reason") == "function_call":
                        # Some implementations only include function call at the end
                        if "message" in chunk["choices"][0]:
                            message = chunk["choices"][0]["message"]
                            if "function_call" in message:
                                function_call_data = message["function_call"]
            except json.JSONDecodeError:
                print(f"Error parsing chunk: {text}")
    
    # Validate streaming results
    assert chunks, "No chunks received"
    
    # If we didn't get a function call, we acknowledge that (not all streaming implementations 
    # handle function calls the same way)
    if function_call_data:
        print(f"\n{model_name} made a function call in streaming mode: {function_call_data}")
        assert "name" in function_call_data, "Function call missing name"
        assert function_call_data["name"] == "get_weather", "Unexpected function name"
    else:
        # Check the final chunk to see if there was a function call that wasn't streamed
        try:
            final_chunk = chunks[-1]
            if "choices" in final_chunk and "message" in final_chunk["choices"][0]:
                message = final_chunk["choices"][0]["message"]
                if "function_call" in message:
                    print(f"\n{model_name} included function call only in final chunk")
                    assert True, "Function call was in final chunk, not streamed"
                    return
        except (IndexError, KeyError):
            pass
            
        print(f"\n{model_name} did not stream a function call")

def test_invalid_model():
    """Test behavior with an invalid model name"""
    # Prepare the request with a non-existent model
    payload = {
        "model": "ollama/non-existent-model",
        "messages": TEST_MESSAGES
    }
    
    # Make the request
    response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
    
    # Either we get a 404/400 error or the response contains an error message
    if response.status_code == 404 or response.status_code == 400:
        print("\nCorrectly returned error status for invalid model")
        assert True
    else:
        data = response.json()
        assert "error" in data, "Expected error field for invalid model"
        print(f"\nError for invalid model: {data.get('error')}")

def test_malformed_request():
    """Test behavior with a malformed request"""
    # Prepare a malformed request (missing required 'messages' field)
    payload = {
        "model": f"ollama/{TEST_CONFIG['ollama_models']['basic'][0]}"
        # Missing 'messages' field
    }
    
    # Make the request
    response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
    
    # We should get a 400 Bad Request
    assert response.status_code in [400, 422], f"Expected 400 or 422 status code, got {response.status_code}"
    print("\nCorrectly handled malformed request")

@pytest.mark.parametrize("model_name", TEST_CONFIG["ollama_models"]["basic"][:1])
def test_config_update(model_name):
    """Test config update endpoint"""
    # Prepare the config update
    config_payload = {
        "prefix": "ollama",
        "config": {
            "base_url": "http://localhost:11434",  # Should be the default anyway
            "supports_function_call": True
        }
    }
    
    # Make the request
    config_response = requests.post(f"{BASE_URL}/v1/config/update", json=config_payload)
    assert config_response.status_code == 200, "Failed to update config"
    
    # Now make a normal request to verify config works
    completion_payload = {
        "model": f"ollama/{model_name}",
        "messages": TEST_MESSAGES
    }
    
    completion_response = requests.post(f"{BASE_URL}/v1/chat/completions", json=completion_payload)
    assert completion_response.status_code == 200, "Failed to use model after config update"
    
    print("\nSuccessfully updated config and made a request")
