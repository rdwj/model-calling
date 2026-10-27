# Testing Guide

This guide covers how to use the Model Calling test suite to verify functionality with local Ollama models, with a particular focus on testing function calling (tool use) capabilities.

## Running Tests

Model Calling includes several test scripts for convenience:

```bash
# Make scripts executable first
chmod +x run_tests.sh test_ollama.sh

# Run all tests
./run_tests.sh

# Run only function calling tests
./test_ollama.sh function

# Run only basic tests
./test_ollama.sh basic

# Run a quick verification
./test_ollama.sh quick
```

## Auto-Configuring Tests

For the easiest setup, you can use the auto-configuration script which automatically detects all your locally installed Ollama models and configures the tests to use them:

```bash
# Make script executable
chmod +x auto_config_tests.sh

# Run auto-configuration
./auto_config_tests.sh
```

This script will:
1. Run `ollama list` to find all available models
2. Identify which models are likely to support function calling
3. Create a configuration file with the detected models
4. Update the test configuration to use these models

After running this script, the tests will automatically use your available models, with models that likely support function calling being used for function calling tests.

## What Gets Tested

The test suite verifies several key aspects of the Model Calling service:

### Basic Functionality
- Service health and endpoints
- Model listing
- Basic completions with different models
- System message handling
- Parameter handling (temperature, etc.)

### Function Calling / Tool Use
- Weather function identification and parameter extraction
- Calculator function calling
- Multiple tool selection
- Complete function flow (request → function call → result → response)

### Streaming and Error Handling
- Streaming text completions
- Streaming with function calls
- Error handling for invalid models
- Error handling for malformed requests
- Configuration updates

## Configuring Test Models

By default, the tests use models that are likely to be available locally through Ollama:

```python
TEST_CONFIG = {
    "ollama_models": {
        # Models that support function calling
        "function_calling": FUNCTION_CALLING_MODELS[:1],  # Only use the first model to speed up tests
        # Models for basic tests
        "basic": BASIC_MODELS[:1]  # Only use the first model to speed up tests
    }
}
```

You can customize which models to test by:
1. Running the auto-configuration script (`./auto_config_tests.sh`)
2. Manually editing the `conftest.py` file

## Understanding Test Output

The test output shows detailed information about each test:

```
test_weather_function_call[mistral-small3.1:24b] 
mistral-small3.1:24b successfully called get_weather function with location: Tokyo
PASSED
```

If a test fails, it will provide information about what went wrong:

```
test_weather_function_call[mistral-small3.1:24b] 
FAILED
AssertionError: Function name mismatch: Expected get_weather, got get_current_weather
```

## Testing in Different Environments

To test against a Model Calling service running on a different URL:

```bash
# Set the service URL
export MODEL_CALLING_URL="http://your-service-url:8000"

# Run tests
./run_tests.sh
```

## Writing Custom Tests

You can add your own tests by creating new test files in the `tests` directory. See the [tests README](tests/README.md) for more details on the test structure and how to add new tests.

## Interpreting Function Call Results

When testing function calling, there are a few possible outcomes:

1. **Function Call Success**: The model correctly identifies the need to call a function and provides appropriate arguments.

2. **Text Response Instead**: Some models might respond with text instead of a function call. This is noted in the test output but not considered a failure, as models have different capabilities.

3. **Incorrect Function Selection**: If multiple tools are available and the model chooses the wrong one, this is reported as a test failure.

4. **Malformed Arguments**: If the model makes a function call but provides invalid arguments, this is reported as a test failure.

## Best Practices

1. **Start Local Service First**: Always make sure the Model Calling service is running before running tests.

2. **Start with Quick Test**: Use `./test_ollama.sh quick` to verify basic functionality before running all tests.

3. **Test One Category at a Time**: Use the category-specific test commands to focus on particular functionality.

4. **Check Model Availability**: Ensure the models listed in `conftest.py` are available in your Ollama installation.

## Diagnosing Model Issues

If you encounter issues with the tests, you can use the diagnosis script to check which models are working with your Model Calling service:

```bash
# Make script executable
chmod +x diagnose_models.sh

# Run diagnosis
./diagnose_models.sh
```

This script will:
1. Check if the service is running
2. Test the model listing endpoint
3. Test several models to see which ones work
4. Test function calling capabilities

Based on the results, you can update the test configuration in `tests/conftest.py` to use models that are compatible with your setup.
