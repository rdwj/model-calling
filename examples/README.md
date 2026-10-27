# Model Calling Examples

This directory contains example scripts demonstrating how to use the model-calling service.

## Examples

1. **ollama_example.py** - Basic examples using Ollama models with both standard chat and function calling
2. **vllm_example.py** - Examples of using vLLM-deployed models on remote servers
3. **agent_example.py** - How to build a complete agent that uses function calling to retrieve data
4. **library_usage.py** - How to use model-calling as an imported library in your own applications
5. **hosted_providers.py** - Examples of using hosted providers like OpenAI, Anthropic, and Cohere
6. **vision_example.py** - Examples of using vision capabilities for image analysis and alt text generation
7. **reasoning_example.py** - Examples of using reasoning capabilities for solving problems step-by-step

## Running the Examples

Make sure the model-calling service is running first:

```bash
cd /Users/wesjackson/Code/model-calling
python -m model_calling.server
```

Then, in a separate terminal, run any of the examples:

```bash
cd /Users/wesjackson/Code/model-calling
python -m examples.ollama_example
```

## Using Different Models

The examples are configured to use `mistral-small3.1:24b` by default. You can modify the model name in the examples to use any Ollama or vLLM model you have available.

For Ollama models, use the format:
```
ollama/model-name
```

For vLLM models, use the format:
```
vllm/model-name
```

## Function Calling

Not all models support function calling. The model-calling service tries to detect function-calling capable models based on their names, but you may need to manually configure this for some models.

You can update the configuration for a specific model type using the client:

```python
client.update_config(
    prefix="ollama",
    config={"supports_function_call": True}
)
```

## Using Hosted Providers

To run the hosted providers example, you'll need API keys for the services you want to test:

1. Copy the example .env file and add your API keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

2. Run the example:

```bash
python -m examples.hosted_providers
```

The example will skip any providers for which you don't have API keys.
