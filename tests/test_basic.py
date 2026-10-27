"""
Tests for basic Model Calling functionality with Ollama models.
"""

import pytest
import requests
import json
from tests.conftest import TEST_CONFIG, TEST_MESSAGES, TEST_SYSTEM_MESSAGES

# Base URL for API calls
BASE_URL = TEST_CONFIG["service_url"]


def test_service_health():
    """Test that the service is running"""
    # Make a request to the root endpoint
    response = requests.get(f"{BASE_URL}")
    assert response.status_code == 200, "Service is not running"


def test_model_list():
    """Test the models endpoint returns a list of models"""
    response = requests.get(f"{BASE_URL}/v1/models")
    assert response.status_code == 200, "Failed to get models list"

    data = response.json()
    assert "data" in data, "Response does not contain 'data' field"
    assert isinstance(data["data"], list), "'data' field is not a list"

    # Check for at least one model
    assert len(data["data"]) > 0, "No models found"

    # Check model format
    model = data["data"][0]
    assert "id" in model, "Model does not have 'id' field"


@pytest.mark.parametrize("model_name", TEST_CONFIG["ollama_models"]["basic"])
def test_basic_completion(model_name):
    """Test basic text completion with different models"""
    # Prepare the request
    payload = {"model": f"ollama/{model_name}", "messages": TEST_MESSAGES}

    # Make the request
    response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
    assert response.status_code == 200, f"Failed to get completion from {model_name}"

    # Validate response format
    data = response.json()
    assert "choices" in data, "Response does not contain 'choices'"
    assert len(data["choices"]) > 0, "No choices returned"
    assert "message" in data["choices"][0], "No message in first choice"
    assert "content" in data["choices"][0]["message"], "No content in message"

    # Check content is not empty
    content = data["choices"][0]["message"]["content"]
    assert content.strip(), "Content is empty"

    # Print for manual verification
    print(f"\nResponse from {model_name}: {content[:100]}...")


@pytest.mark.parametrize("model_name", TEST_CONFIG["ollama_models"]["basic"])
def test_system_message(model_name):
    """Test using a system message"""
    # Prepare the request
    payload = {"model": f"ollama/{model_name}", "messages": TEST_SYSTEM_MESSAGES}

    # Make the request
    response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
    assert (
        response.status_code == 200
    ), f"Failed to get completion with system message from {model_name}"

    # Validate content
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    assert content.strip(), "Content is empty"

    # Print for manual verification
    print(f"\nResponse with system message from {model_name}: {content[:100]}...")


@pytest.mark.parametrize("model_name", TEST_CONFIG["ollama_models"]["basic"])
def test_temperature_param(model_name):
    """Test temperature parameter"""
    # Prepare requests with different temperatures
    payload_low_temp = {
        "model": f"ollama/{model_name}",
        "messages": TEST_MESSAGES,
        "temperature": 0.1,
    }

    payload_high_temp = {
        "model": f"ollama/{model_name}",
        "messages": TEST_MESSAGES,
        "temperature": 0.9,
    }

    # Make requests
    response_low = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload_low_temp)
    response_high = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload_high_temp)

    assert response_low.status_code == 200, "Failed with low temperature"
    assert response_high.status_code == 200, "Failed with high temperature"

    # Validate responses (can't check temperature effect directly,
    # but can ensure it accepts and processes the parameter)
    data_low = response_low.json()
    data_high = response_high.json()

    assert "choices" in data_low, "Low temp response missing choices"
    assert "choices" in data_high, "High temp response missing choices"
