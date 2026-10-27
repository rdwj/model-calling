"""
Tests for function calling (tool use) capabilities with Ollama models.
"""

import pytest
import requests
import json
from tests.integration.conftest import (
    TEST_CONFIG,
    WEATHER_TOOL,
    CALCULATOR_TOOL,
    MULTIPLE_TOOLS,
    get_weather,
    calculate,
)

pytestmark = pytest.mark.integration

# Base URL for API calls
BASE_URL = TEST_CONFIG["service_url"]


@pytest.mark.parametrize("model_name", TEST_CONFIG["ollama_models"]["function_calling"])
def test_weather_function_call(model_name):
    """Test function calling with weather tool"""
    # Prepare the request to trigger a function call
    payload = {
        "model": f"ollama/{model_name}",
        "messages": [{"role": "user", "content": "What's the weather like in Tokyo?"}],
        "tools": WEATHER_TOOL,
    }

    # Make the request
    response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
    assert response.status_code == 200, f"Failed to get completion from {model_name}"

    # Validate response format
    data = response.json()
    assert "choices" in data, "Response does not contain 'choices'"
    assert len(data["choices"]) > 0, "No choices returned"
    assert "message" in data["choices"][0], "No message in first choice"

    message = data["choices"][0]["message"]

    # Check if function call was requested
    if "function_call" in message:
        assert "name" in message["function_call"], "No function name in function_call"
        assert "arguments" in message["function_call"], "No arguments in function_call"

        # Validate function call details
        function_name = message["function_call"]["name"]
        assert function_name == "get_weather", f"Expected get_weather, got {function_name}"

        # Parse arguments
        arguments = json.loads(message["function_call"]["arguments"])
        assert "location" in arguments, "No location in arguments"
        assert arguments["location"].lower() in [
            "tokyo",
            "tokyo, japan",
        ], f"Unexpected location: {arguments['location']}"

        print(
            f"\n{model_name} successfully called get_weather function with location: {arguments['location']}"
        )
        return True
    else:
        # Some models might respond without function call, which is acceptable but noted
        print(
            f"\n{model_name} did not call the function, instead responded with: {message['content'][:100]}..."
        )
        return False


@pytest.mark.parametrize("model_name", TEST_CONFIG["ollama_models"]["function_calling"])
def test_calculator_function_call(model_name):
    """Test function calling with calculator tool"""
    # Prepare the request to trigger a function call
    payload = {
        "model": f"ollama/{model_name}",
        "messages": [{"role": "user", "content": "Calculate 123 * 456"}],
        "tools": CALCULATOR_TOOL,
    }

    # Make the request
    response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
    assert response.status_code == 200, f"Failed to get completion from {model_name}"

    # Validate response
    data = response.json()
    message = data["choices"][0]["message"]

    # Check if function call was requested
    if "function_call" in message:
        function_name = message["function_call"]["name"]
        assert function_name == "calculate", f"Expected calculate, got {function_name}"

        # Parse arguments
        arguments = json.loads(message["function_call"]["arguments"])
        assert "expression" in arguments, "No expression in arguments"

        # Evaluate expression to verify it's sensible
        result = calculate(arguments["expression"])
        print(f"\n{model_name} called calculate with: {arguments['expression']}, result: {result}")
        return True
    else:
        print(
            f"\n{model_name} did not call the calculator, instead responded with: {message['content'][:100]}..."
        )
        return False


@pytest.mark.parametrize("model_name", TEST_CONFIG["ollama_models"]["function_calling"])
def test_multiple_tools(model_name):
    """Test offering multiple tools and checking selection"""
    # Prepare the request to trigger a weather function call when multiple tools are available
    payload = {
        "model": f"ollama/{model_name}",
        "messages": [{"role": "user", "content": "What's the weather like in Paris?"}],
        "tools": MULTIPLE_TOOLS,
    }

    # Make the request
    response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
    assert response.status_code == 200, f"Failed to get completion from {model_name}"

    # Validate response
    data = response.json()
    message = data["choices"][0]["message"]

    # Check if function call was requested
    if "function_call" in message:
        function_name = message["function_call"]["name"]
        assert function_name == "get_weather", f"Expected get_weather, got {function_name}"

        # Parse arguments
        arguments = json.loads(message["function_call"]["arguments"])
        assert "location" in arguments, "No location in arguments"
        assert arguments["location"].lower() in [
            "paris",
            "paris, france",
        ], f"Unexpected location: {arguments['location']}"

        print(
            f"\n{model_name} correctly selected get_weather from multiple tools with location: {arguments['location']}"
        )
        return True
    else:
        print(
            f"\n{model_name} did not call any function, instead responded with: {message['content'][:100]}..."
        )
        return False


@pytest.mark.parametrize("model_name", TEST_CONFIG["ollama_models"]["function_calling"])
def test_complete_function_flow(model_name):
    """Test complete function calling flow with function result handling"""
    # Step 1: Make initial request to get function call
    payload = {
        "model": f"ollama/{model_name}",
        "messages": [{"role": "user", "content": "What's the weather like in Berlin?"}],
        "tools": WEATHER_TOOL,
    }

    # Make the request
    response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
    assert response.status_code == 200, f"Failed to get initial completion from {model_name}"

    # Process response
    data = response.json()
    message = data["choices"][0]["message"]

    # If no function call, skip the rest of the test
    if "function_call" not in message:
        pytest.skip(f"{model_name} did not call a function")

    # Extract function call
    function_name = message["function_call"]["name"]
    arguments = json.loads(message["function_call"]["arguments"])

    # Step 2: Simulate function execution
    weather_data = get_weather(arguments["location"])

    # Step 3: Continue the conversation with function result
    function_payload = {
        "model": f"ollama/{model_name}",
        "messages": [
            {"role": "user", "content": "What's the weather like in Berlin?"},
            {"role": "assistant", "content": "", "function_call": message["function_call"]},
            {"role": "function", "name": function_name, "content": json.dumps(weather_data)},
        ],
    }

    # Make the follow-up request
    function_response = requests.post(f"{BASE_URL}/v1/chat/completions", json=function_payload)
    assert (
        function_response.status_code == 200
    ), f"Failed to get function result completion from {model_name}"

    # Validate final response
    final_data = function_response.json()
    final_message = final_data["choices"][0]["message"]

    # Verify final response contains weather information
    assert "content" in final_message, "No content in final message"
    assert final_message["content"].strip(), "Final content is empty"

    # Check for expected content elements
    content = final_message["content"].lower()
    assert any(
        word in content for word in ["berlin", "weather", "temperature", "degrees", "celsius"]
    ), "Final response doesn't mention weather information"

    print(f"\nComplete function flow with {model_name} successful:")
    print(f"Input → Function Call → Function Result → Final Response:")
    print(f"Final: {final_message['content'][:150]}...")

    return True
