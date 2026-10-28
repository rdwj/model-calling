# Migration Guide: From Local Development to Production

This guide covers the transition from local development with Ollama to production deployment in environments like OpenShift, addressing common challenges such as authentication, service URLs, and ensuring code compatibility.

- [Overview](#overview)
- [Migration Scenarios](#migration-scenarios)
- [Authentication Changes](#authentication-changes)
- [Configuration Management](#configuration-management)
- [OpenShift-Specific Considerations](#openshift-specific-considerations)
- [Complete Examples](#complete-examples)
- [Troubleshooting](#troubleshooting)

## Overview

A common development workflow involves:

1. Developing locally with Ollama for fast iteration and no authentication
2. Testing with hosted services like Anthropic or OpenAI
3. Deploying to production using vLLM or other enterprise-grade services on OpenShift

Model Calling is designed to make this transition as seamless as possible, requiring minimal code changes when moving between environments.

## Migration Scenarios

### Scenario 1: Ollama → vLLM on OpenShift

**What changes:**
- Service URL
- Provider prefix (`ollama/` → `vllm/`)
- Authentication (none → token-based)

**What stays the same:**
- Message format
- Tool/function definitions
- Response processing logic
- Agent business logic

### Scenario 2: Ollama → Hosted API (Anthropic/OpenAI)

**What changes:**
- Service URL
- Provider prefix (`ollama/` → `anthropic/` or `openai/`)
- Authentication (none → API key)
- Possibly the model name

**What stays the same:**
- Message format
- Tool/function definitions
- Response processing logic

### Scenario 3: Multiple Environments (Dev/Test/Prod)

**What changes:**
- Service URLs
- Provider prefixes
- Authentication methods
- Possibly model names

**The challenge:** Managing different configurations without code changes

## Authentication Changes

### From Unauthenticated (Ollama) to Authenticated Services

#### Local Development (No Authentication)

```python
from model_calling.client import SyncModelCallingClient

# Simple unauthenticated local setup
client = SyncModelCallingClient()  # Default localhost:8000

response = client.chat_completion(
    model="ollama/mistral-small3.1:24b",
    messages=[
        {"role": "user", "content": "Generate a product description for a coffee maker."}
    ]
)
```

#### Production with Authentication

There are several approaches to add authentication:

##### 1. Environment Variables (Recommended)

```python
# Set in environment or .env file:
# ANTHROPIC_API_KEY=your_key
# VLLM_API_KEY=your_token

from model_calling.client import SyncModelCallingClient

# The base URL changes, but code structure remains identical
client = SyncModelCallingClient(base_url="https://model-calling.openshift-cluster.com")

response = client.chat_completion(
    model="vllm/mistral-7b",  # Just change the provider prefix
    messages=[
        {"role": "user", "content": "Generate a product description for a coffee maker."}
    ]
)
```

##### 2. Explicit Configuration

```python
from model_calling.client import SyncModelCallingClient

client = SyncModelCallingClient(base_url="https://model-calling.openshift-cluster.com")

# Configure authentication explicitly
client.update_config(
    prefix="vllm",
    config={
        "api_key": "your_vllm_token",
        "base_url": "https://vllm-service.openshift-cluster.com/v1",
        "headers": {  # Optional custom headers
            "X-Project-ID": "your-project-id"
        }
    }
)

# Same code as before
response = client.chat_completion(
    model="vllm/mistral-7b",
    messages=[
        {"role": "user", "content": "Generate a product description for a coffee maker."}
    ]
)
```

### Authentication for Different Providers

#### vLLM on OpenShift

```python
client.update_config(
    prefix="vllm",
    config={
        "api_key": "your_vllm_token",
        "base_url": "https://vllm-service.openshift-cluster.com/v1",
        "headers": {
            "Authorization": f"Bearer {vllm_token}"
        }
    }
)
```

#### Anthropic

```python
client.update_config(
    prefix="anthropic",
    config={
        "api_key": "your_anthropic_api_key",
        "base_url": "https://api.anthropic.com/v1/messages"
    }
)
```

#### OpenAI

```python
client.update_config(
    prefix="openai",
    config={
        "api_key": "your_openai_api_key",
        "org_id": "your_openai_org_id",  # Optional
        "base_url": "https://api.openai.com/v1"
    }
)
```

For `vllm`, `openai`, and `openrouter`, `base_url` is the API root with the
`/v1` version included, as in the official `openai` SDK. The gateway appends
`/chat/completions`.

#### OpenRouter

`openrouter/` models use `https://openrouter.ai/api/v1` and read
`OPENROUTER_API_KEY` from the environment by default:

```python
response = client.chat_completion(
    model="openrouter/mistralai/mistral-nemo",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Configuration Management

### Environment-Based Configuration

```python
import os
from model_calling.client import SyncModelCallingClient

# Determine environment
ENV = os.environ.get("DEPLOYMENT_ENV", "development")

# Configure based on environment
if ENV == "development":
    base_url = "http://localhost:8000"
    model_name = "ollama/mistral-small3.1:24b"
elif ENV == "staging":
    base_url = "https://model-calling-staging.example.com"
    model_name = "vllm/mistral-7b"
elif ENV == "production":
    base_url = "https://model-calling-prod.openshift-cluster.com"
    model_name = "vllm/mistral-7b"

# Create client with appropriate base URL
client = SyncModelCallingClient(base_url=base_url)

# Use the environment-appropriate model
response = client.chat_completion(
    model=model_name,
    messages=[
        {"role": "user", "content": "Tell me about AI."}
    ]
)
```

### Configuration File Approach

```python
import yaml
import os
from model_calling.client import SyncModelCallingClient

# Load configuration from file
env = os.environ.get("DEPLOYMENT_ENV", "development")
config_path = f"config/{env}.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Create client with configuration
client = SyncModelCallingClient(base_url=config["base_url"])

# Configure providers if needed
for provider, provider_config in config["providers"].items():
    client.update_config(prefix=provider, config=provider_config)

# Use configured model
response = client.chat_completion(
    model=config["default_model"],
    messages=[
        {"role": "user", "content": "Tell me about AI."}
    ]
)
```

Example `config/development.yaml`:
```yaml
base_url: http://localhost:8000
default_model: ollama/mistral-small3.1:24b
providers:
  ollama:
    base_url: http://localhost:11434
```

Example `config/production.yaml`:
```yaml
base_url: https://model-calling-prod.openshift-cluster.com
default_model: vllm/mistral-7b
providers:
  vllm:
    base_url: https://vllm-service.openshift-cluster.com/v1
    api_key: ${VLLM_API_KEY}  # Replaced at runtime
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
```

## OpenShift-Specific Considerations

### Using Kubernetes Secrets

Store your API keys and tokens securely:

```yaml
# kubernetes-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: model-api-keys
type: Opaque
data:
  ANTHROPIC_API_KEY: base64_encoded_key_here
  VLLM_API_KEY: base64_encoded_token_here
```

Mount as environment variables:

```yaml
# deployment.yaml
spec:
  containers:
  - name: your-app
    env:
      - name: ANTHROPIC_API_KEY
        valueFrom:
          secretKeyRef:
            name: model-api-keys
            key: ANTHROPIC_API_KEY
      - name: VLLM_API_KEY
        valueFrom:
          secretKeyRef:
            name: model-api-keys
            key: VLLM_API_KEY
```

### Configuring vLLM on OpenShift

When deploying vLLM on OpenShift, you may need to set up:

1. **Authentication Middleware**: Typically using Bearer tokens
2. **CORS Configuration**: If accessing from web applications 
3. **Route Configuration**: Setting up proper routes for the service

Model Calling can be configured to work with these setups:

```python
client.update_config(
    prefix="vllm",
    config={
        "base_url": "https://vllm-route.openshift-apps.example.com/v1",
        "headers": {
            "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY')}",
            "X-Request-ID": "unique-request-id"  # Optional for tracking
        }
    }
)
```

### Working with Chat Templates in vLLM

If your vLLM deployment uses a default chat template, Model Calling works seamlessly with it:

```python
# vLLM with a default chat template (e.g., for Mistral)
client.update_config(
    prefix="vllm",
    config={
        "base_url": "https://vllm-service.openshift-cluster.com/v1",
        "api_key": os.environ.get("VLLM_API_KEY")
    }
)

# Your code remains unchanged
response = client.chat_completion(
    model="vllm/mistral-7b",
    messages=[...],
    tools=[...]
)
```

vLLM is reached through its OpenAI-compatible `/v1/chat/completions` endpoint, so the server applies the model's chat template. The legacy `/generate` prompt mode is no longer supported.

## Complete Examples

### Complete Migration Example: Local to OpenShift

Here's a complete example of an agent migrating from local Ollama to vLLM on OpenShift:

#### Local Development Version

```python
import json
from model_calling.client import SyncModelCallingClient

# Simple weather function
def get_weather(location):
    return {"temperature": 22, "unit": "celsius", "description": "Sunny"}

# Define tools
weather_tool = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

def run_weather_agent(query):
    # Create client - local development with Ollama
    client = SyncModelCallingClient()
    
    try:
        # Make initial request
        response = client.chat_completion(
            model="ollama/granite:34b",  # Local Granite model
            messages=[
                {"role": "user", "content": query}
            ],
            tools=weather_tool
        )
        
        # Process function calls
        message = response["choices"][0]["message"]
        if "function_call" in message:
            # Extract function call details
            function_name = message["function_call"]["name"]
            arguments = json.loads(message["function_call"]["arguments"])
            
            # Call the appropriate function
            if function_name == "get_weather":
                weather_data = get_weather(arguments["location"])
            else:
                weather_data = {"error": "Unknown function"}
            
            # Continue the conversation with the function result
            final_response = client.chat_completion(
                model="ollama/granite:34b",
                messages=[
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": "", "function_call": message["function_call"]},
                    {"role": "function", "name": function_name, "content": json.dumps(weather_data)}
                ]
            )
            
            return final_response["choices"][0]["message"]["content"]
        else:
            return message["content"]
    
    finally:
        client.close()

# Use the agent
result = run_weather_agent("What's the weather like in Paris?")
print(result)
```

#### Production Version for OpenShift

```python
import json
import os
from model_calling.client import SyncModelCallingClient

# Simple weather function
def get_weather(location):
    return {"temperature": 22, "unit": "celsius", "description": "Sunny"}

# Define tools - UNCHANGED
weather_tool = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

def run_weather_agent(query):
    # Get configuration from environment
    model_calling_url = os.environ.get("MODEL_CALLING_URL", "https://model-calling.openshift-cluster.com")
    vllm_api_key = os.environ.get("VLLM_API_KEY", "")
    model_name = os.environ.get("MODEL_NAME", "vllm/granite:34b")  # Default to vLLM
    
    # Create client with production URL
    client = SyncModelCallingClient(base_url=model_calling_url)
    
    # Configure vLLM authentication if needed
    if vllm_api_key and model_name.startswith("vllm/"):
        client.update_config(
            prefix="vllm",
            config={"api_key": vllm_api_key}
        )
    
    try:
        # Make initial request - ONLY MODEL NAME CHANGED
        response = client.chat_completion(
            model=model_name,  # Now using vLLM on OpenShift
            messages=[
                {"role": "user", "content": query}
            ],
            tools=weather_tool
        )
        
        # Process function calls - COMPLETELY UNCHANGED
        message = response["choices"][0]["message"]
        if "function_call" in message:
            # Extract function call details
            function_name = message["function_call"]["name"]
            arguments = json.loads(message["function_call"]["arguments"])
            
            # Call the appropriate function
            if function_name == "get_weather":
                weather_data = get_weather(arguments["location"])
            else:
                weather_data = {"error": "Unknown function"}
            
            # Continue the conversation with the function result
            final_response = client.chat_completion(
                model=model_name,  # Using the environment-configured model
                messages=[
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": "", "function_call": message["function_call"]},
                    {"role": "function", "name": function_name, "content": json.dumps(weather_data)}
                ]
            )
            
            return final_response["choices"][0]["message"]["content"]
        else:
            return message["content"]
    
    finally:
        client.close()

# Use the agent - UNCHANGED
result = run_weather_agent("What's the weather like in Paris?")
print(result)
```

### Hybrid Environment Example

Here's an example that can dynamically select the best available model across multiple environments:

```python
import os
import json
from model_calling.client import SyncModelCallingClient

class ModelSelector:
    """Select the best model based on availability and environment"""
    
    def __init__(self):
        # Create client
        model_calling_url = os.environ.get("MODEL_CALLING_URL", "http://localhost:8000")
        self.client = SyncModelCallingClient(base_url=model_calling_url)
        
        # Configure available models
        self.configure_models()
        
    def configure_models(self):
        """Configure models based on environment variables"""
        self.models = []
        
        # Add vLLM model if available
        vllm_api_key = os.environ.get("VLLM_API_KEY")
        if vllm_api_key:
            self.client.update_config(
                prefix="vllm",
                config={"api_key": vllm_api_key}
            )
            self.models.append("vllm/mistral-7b")
        
        # Add Anthropic model if available
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            self.models.append("anthropic/claude-3-sonnet-20240229")
        
        # Ollama is always available as fallback in development
        self.models.append("ollama/mistral-small3.1:24b")
    
    def get_best_model(self):
        """Get the best available model"""
        if not self.models:
            raise ValueError("No models configured")
        return self.models[0]
    
    def close(self):
        """Close the client"""
        self.client.close()

# Example agent that works across environments
class FlexibleAgent:
    """Agent that works across multiple environments"""
    
    def __init__(self):
        self.model_selector = ModelSelector()
        self.client = self.model_selector.client
        
        # Define tools
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather in a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
    
    def run(self, query):
        """Run the agent with the best available model"""
        try:
            # Get the best model
            model = self.model_selector.get_best_model()
            print(f"Using model: {model}")
            
            # Make initial request
            response = self.client.chat_completion(
                model=model,
                messages=[
                    {"role": "user", "content": query}
                ],
                tools=self.tools
            )
            
            # Process function calls
            message = response["choices"][0]["message"]
            if "function_call" in message:
                # ... function call handling (omitted for brevity)
                # This would be the same regardless of model
                pass
            
            return message["content"]
        
        finally:
            pass  # Don't close client here
    
    def close(self):
        """Clean up resources"""
        self.model_selector.close()

# Example usage
agent = FlexibleAgent()
try:
    result = agent.run("Tell me about AI")
    print(result)
finally:
    agent.close()
```

## Troubleshooting

### Common Migration Issues

#### Authentication Errors

**Symptoms:**
- `401 Unauthorized` errors
- `Authentication failed` messages

**Solutions:**
- Check that the correct API key or token is provided
- Verify the authentication header format (e.g., `Bearer` prefix)
- Ensure secrets are properly mounted in Kubernetes/OpenShift

#### Model Not Found

**Symptoms:**
- `Model not found` errors
- `Unknown model` messages

**Solutions:**
- Verify the model name is correct for the target environment
- Ensure the model is actually deployed on the target vLLM instance
- Check provider prefix (`vllm/` vs `ollama/`)

#### Function Calling Issues

**Symptoms:**
- Function calling not working in production, but works locally
- Model not returning function calls in expected format

**Solutions:**
- Verify the target model supports function calling
- Update configuration to specify function call support:
  ```python
  client.update_config(
      prefix="vllm",
      config={"supports_function_call": True}
  )
  ```
- Check if the vLLM deployment has the right chat template

### Testing Across Environments

A good practice is to create a simple test script that checks basic connectivity and capabilities:

```python
import os
import json
from model_calling.client import SyncModelCallingClient

def test_model_calling():
    """Test Model Calling across environments"""
    # Get configuration from environment
    model_calling_url = os.environ.get("MODEL_CALLING_URL", "http://localhost:8000")
    model_name = os.environ.get("MODEL_NAME", "ollama/mistral-small3.1:24b")
    
    print(f"Testing with: URL={model_calling_url}, Model={model_name}")
    
    client = SyncModelCallingClient(base_url=model_calling_url)
    
    try:
        # Basic text completion
        print("Testing basic completion...")
        response = client.chat_completion(
            model=model_name,
            messages=[
                {"role": "user", "content": "Say hello"}
            ]
        )
        
        print(f"Response: {response['choices'][0]['message']['content']}")
        
        # Function calling
        print("Testing function calling...")
        response = client.chat_completion(
            model=model_name,
            messages=[
                {"role": "user", "content": "What's the weather in Tokyo?"}
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather in a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City name"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                }
            ]
        )
        
        message = response["choices"][0]["message"]
        if "function_call" in message:
            print(f"Function call: {message['function_call']['name']}")
            print(f"Arguments: {message['function_call']['arguments']}")
        else:
            print(f"No function call. Response: {message['content']}")
        
        print("All tests passed!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        client.close()

if __name__ == "__main__":
    test_model_calling()
```

This script can be run in different environments to verify that your Model Calling setup is working as expected.

## Conclusion

The migration from local development to production should require minimal code changes when using Model Calling. By following the patterns in this guide, you can build applications that seamlessly work across environments, from local Ollama development to enterprise-grade vLLM deployments on OpenShift.

Remember these key principles:

1. **Externalize Configuration**: Use environment variables or config files for environment-specific settings
2. **Maintain Abstraction**: Let Model Calling handle the provider-specific details
3. **Test Across Environments**: Verify functionality works consistently in all target environments

With this approach, your applications can develop locally with fast iteration and deploy to production with confidence.
