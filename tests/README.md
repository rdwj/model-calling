# Model Calling Tests

This directory contains tests for the Model Calling service, focusing on verifying local Ollama model functionality and especially function calling (tool use) capabilities.

## Test Structure

- **conftest.py** - Common test utilities and configurations
- **pytest.ini** - Pytest configuration
- **test_basic.py** - Tests for basic functionality
- **test_function_calling.py** - Tests for function calling/tool use
- **test_streaming_and_errors.py** - Tests for streaming responses and error handling

## Running Tests

From the project root directory:

```bash
# Run all tests
./run_tests.sh

# Run specific test categories
./test_ollama.sh function    # Function calling tests
./test_ollama.sh basic       # Basic tests
./test_ollama.sh streaming   # Streaming tests
./test_ollama.sh errors      # Error handling tests
./test_ollama.sh quick       # Quick verification
```

## Configuring Tests

You can modify which models are tested by editing the `TEST_CONFIG` dictionary in `conftest.py`:

```python
TEST_CONFIG = {
    "service_url": os.environ.get("MODEL_CALLING_URL", "http://localhost:8000"),
    "ollama_models": {
        # Models that support function calling
        "function_calling": [
            "mistral-small3.1:24b",  # Edit this list to include your models
            "granite3-dense:8b",
            "llama3:latest"
        ][:1],  # Limit to first model for faster tests
        # Models that may not support function calling
        "basic": [
            "tinyllama:latest"
        ]
    }
}
```

## Adding New Tests

To add new tests:

1. Create a new test file named `test_*.py`
2. Import necessary utilities from `conftest`
3. Use pytest fixtures and parameterization for model-specific tests

Example:

```python
import pytest
import requests
from conftest import TEST_CONFIG, TEST_MESSAGES

@pytest.mark.parametrize("model_name", TEST_CONFIG["ollama_models"]["basic"])
def test_my_new_feature(model_name):
    # Test implementation
    pass
```

## Test Categories

### Basic Tests

- Service health check
- Model listing
- Simple completions
- System messages
- Parameter handling (temperature, etc.)

### Function Calling Tests

- Weather function calls
- Calculator function calls
- Multiple tool selection
- Complete function flow

### Streaming and Error Tests

- Streaming responses
- Streaming with function calls
- Error handling (invalid models, malformed requests)
- Configuration updates

## Requirements

- Running Model Calling service (at http://localhost:8000 by default)
- Ollama with appropriate models installed
- Python with pytest installed
