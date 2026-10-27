# Advanced Topics

This page covers advanced topics for working with the Model Calling API.

## Monitoring and Observability

model-calling exposes a callback interface (`model_calling.callbacks.Callback`) with hooks for request start, request end, streaming chunks, errors, and embedding requests. Subclass it and attach instances through an adapter's `callbacks` config to add tracing, logging, or token accounting without changing adapter code:

```python
from model_calling.callbacks import Callback
from model_calling.registry import registry


class UsageLogger(Callback):
    async def on_request_end(self, model, messages, response, **kwargs):
        usage = response.get("usage", {})
        print(f"{model}: {usage.get('total_tokens', 0)} tokens")

    async def on_error(self, model, messages, error, **kwargs):
        print(f"{model} failed: {error}")


registry.update_config("vllm", {"callbacks": [UsageLogger()]})
```

## Creating Custom Adapters

You can extend the system to support new model providers by creating custom adapters.

### Example: Creating a Custom Provider Adapter

```python
# Create a new file: my_provider_adapter.py
import os
import json
import logging
from typing import Dict, Any, List, Optional
import httpx

from model_calling.hosted import HostedProviderAdapter
from model_calling.adapter import create_openai_response, create_openai_streaming_chunk

class MyProviderAdapter(HostedProviderAdapter):
    """Adapter for a custom provider"""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize the adapter with configuration"""
        model_config["provider_name"] = "myprovider"
        super().__init__(model_config)
    
    def _get_api_key(self) -> str:
        """Get API key from environment"""
        return os.environ.get("MY_PROVIDER_API_KEY", "")
    
    def _get_base_url(self) -> str:
        """Get base URL for the API"""
        return os.environ.get("MY_PROVIDER_URL", "https://api.myprovider.com/v1/chat")
    
    async def _make_request_headers(self) -> Dict[str, str]:
        """Create headers for the API request"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    async def translate_request(self, openai_format_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI format to provider format"""
        # Extract messages
        messages = openai_format_request.get("messages", [])
        
        # Create provider-specific request format
        provider_request = {
            "model": self.model_name,
            "messages": messages,  # If provider format matches OpenAI
            "temperature": openai_format_request.get("temperature", 1.0)
        }
        
        # Handle other parameters or conversion as needed
        if "max_tokens" in openai_format_request:
            provider_request["max_output_tokens"] = openai_format_request["max_tokens"]
        
        return provider_request
    
    async def translate_response(self, provider_response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert provider response to OpenAI format"""
        if "error" in provider_response:
            return provider_response
        
        # Extract provider-specific response fields
        content = provider_response.get("output", "")
        
        # Create OpenAI-compatible response
        return create_openai_response(
            model=f"myprovider/{self.model_name}",
            content=content,
            finish_reason="stop"
        )
    
    async def translate_streaming_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a provider streaming chunk to OpenAI format"""
        content = chunk.get("output", "")
        is_final = chunk.get("done", False)
        
        return create_openai_streaming_chunk(
            model=f"myprovider/{self.model_name}",
            content=content,
            is_final=is_final
        )
    
    async def supports_tools(self) -> bool:
        """Check if this model supports tool/function calling"""
        # Update based on provider capabilities
        return False
```

### Registering the Custom Adapter

```python
# Import the adapter
from my_provider_adapter import MyProviderAdapter
from model_calling.registry import registry

# Register the adapter
registry.register_adapter_class("myprovider", MyProviderAdapter)

# Add default configuration
registry.update_config(
    prefix="myprovider",
    config={
        "base_url": "https://api.myprovider.com/v1/chat",
        "api_key": os.environ.get("MY_PROVIDER_API_KEY", "")
    }
)
```

## Performance Optimization

### Connection Pooling

For high-throughput scenarios, you can optimize performance with connection pooling:

```python
import httpx
from model_calling.client import SyncModelCallingClient

# Create a shared httpx client for connection pooling
http_client = httpx.Client(
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20
    ),
    timeout=30.0
)

def get_optimized_client():
    """Create a client with an optimized connection pool"""
    client = SyncModelCallingClient()
    client.client = http_client  # Replace the client with our pooled version
    return client

# Example usage
client = get_optimized_client()
# Use the client as normal
# NOTE: With this approach, you should NOT close the client after each use
```

### Async Batch Processing

For processing many requests efficiently:

```python
import asyncio
import time
from model_calling.client import ModelCallingClient

async def process_batch(queries, model, batch_size=5, rate_limit=10):
    """
    Process queries in batches with rate limiting
    
    Args:
        queries: List of query strings
        model: Model to use
        batch_size: How many queries to process in parallel
        rate_limit: Maximum requests per second
    """
    client = ModelCallingClient()
    results = []
    
    try:
        # Process in batches
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i+batch_size]
            batch_tasks = []
            
            # Create tasks for this batch
            for query in batch:
                task = asyncio.create_task(client.chat_completion(
                    model=model,
                    messages=[{"role": "user", "content": query}]
                ))
                batch_tasks.append(task)
            
            # Wait for all tasks in this batch
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
            
            # Rate limiting
            if i + batch_size < len(queries):
                await asyncio.sleep(batch_size / rate_limit)
        
        return results
    
    finally:
        await client.close()
```

## Containerization

### Podman Compose for Production

Example of a production-oriented `podman-compose.yml`. The `deploy.resources` GPU reservation follows the Compose spec; on Podman you may instead need `--device nvidia.com/gpu=all` via CDI, depending on your setup.

```yaml
services:
  model-calling:
    build:
      context: .
      dockerfile: Containerfile
    restart: always
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
      - LOG_LEVEL=info
      - OLLAMA_BASE_URL=http://ollama:11434
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - COHERE_API_KEY=${COHERE_API_KEY}
    volumes:
      - ./logs:/opt/app-root/src/logs
    networks:
      - model-network
    depends_on:
      - ollama
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s

  ollama:
    image: ollama/ollama:latest
    restart: always
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    networks:
      - model-network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

networks:
  model-network:
    driver: bridge

volumes:
  ollama_data:
```

## Load Balancing

For high-availability setups, you can deploy multiple instances and load balance:

```python
# Example config for a load-balancing client
from model_calling.client import SyncModelCallingClient
import random

class LoadBalancedClient:
    """Client that distributes requests across multiple servers"""
    
    def __init__(self, endpoints):
        """Initialize with multiple endpoints"""
        self.endpoints = endpoints
        self.clients = [SyncModelCallingClient(base_url=endpoint) for endpoint in endpoints]
    
    def chat_completion(self, **kwargs):
        """Send a request to a random endpoint"""
        # Simple round-robin selection - could be improved with health checks
        client = random.choice(self.clients)
        return client.chat_completion(**kwargs)
    
    def close(self):
        """Close all clients"""
        for client in self.clients:
            client.close()

# Example usage
balanced_client = LoadBalancedClient([
    "http://server1:8000",
    "http://server2:8000",
    "http://server3:8000"
])

try:
    response = balanced_client.chat_completion(
        model="ollama/mistral-small3.1:24b",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(response["choices"][0]["message"]["content"])
finally:
    balanced_client.close()
```

## Observability and Monitoring

### Logging Integration

Example of enhanced logging for monitoring:

```python
import logging
import time
import uuid
from model_calling.client import SyncModelCallingClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_requests.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("llm_client")

# Wrapper client with enhanced logging
class MonitoredClient:
    """Client with enhanced logging and monitoring"""
    
    def __init__(self):
        self.client = SyncModelCallingClient()
    
    def chat_completion(self, model, messages, **kwargs):
        """Log request details and timing"""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Log request
        logger.info(f"Request {request_id} - Model: {model} - Start")
        
        try:
            # Make the request
            response = self.client.chat_completion(
                model=model,
                messages=messages,
                **kwargs
            )
            
            # Calculate metrics
            duration = time.time() - start_time
            token_count = len(response["choices"][0]["message"]["content"].split())
            
            # Log success
            logger.info(
                f"Request {request_id} - Model: {model} - Success - "
                f"Duration: {duration:.2f}s - Tokens: {token_count}"
            )
            
            return response
            
        except Exception as e:
            # Log failure
            duration = time.time() - start_time
            logger.error(
                f"Request {request_id} - Model: {model} - Failure - "
                f"Duration: {duration:.2f}s - Error: {str(e)}"
            )
            raise
    
    def close(self):
        self.client.close()
```

### Prometheus Metrics

Example integration with Prometheus metrics:

```python
from prometheus_client import Counter, Histogram, start_http_server
import time
from model_calling.client import SyncModelCallingClient

# Initialize metrics
REQUEST_COUNT = Counter(
    'model_calling_requests_total', 
    'Total number of requests sent',
    ['model', 'status']
)

REQUEST_LATENCY = Histogram(
    'model_calling_request_latency_seconds', 
    'Request latency in seconds',
    ['model']
)

TOKEN_COUNT = Counter(
    'model_calling_tokens_total', 
    'Total number of tokens generated',
    ['model']
)

# Start Prometheus metrics server
start_http_server(8001)

# Client with metrics
class MetricsClient:
    """Client that reports Prometheus metrics"""
    
    def __init__(self):
        self.client = SyncModelCallingClient()
    
    def chat_completion(self, model, messages, **kwargs):
        """Make a request and record metrics"""
        start_time = time.time()
        
        try:
            # Make the request
            response = self.client.chat_completion(
                model=model,
                messages=messages,
                **kwargs
            )
            
            # Record success metrics
            REQUEST_COUNT.labels(model=model, status="success").inc()
            REQUEST_LATENCY.labels(model=model).observe(time.time() - start_time)
            
            # Estimate token count (rough approximation)
            content = response["choices"][0]["message"]["content"]
            token_count = len(content.split()) * 1.3  # rough approximation
            TOKEN_COUNT.labels(model=model).inc(token_count)
            
            return response
            
        except Exception as e:
            # Record failure metrics
            REQUEST_COUNT.labels(model=model, status="failure").inc()
            REQUEST_LATENCY.labels(model=model).observe(time.time() - start_time)
            raise
    
    def close(self):
        self.client.close()
```

## Security and Authentication

### API Key Rotation

Example of periodic API key rotation:

```python
import os
import threading
import time
import random
from model_calling.client import SyncModelCallingClient
from model_calling.registry import registry

class KeyRotationClient:
    """Client that periodically rotates API keys"""
    
    def __init__(self, key_provider, rotation_interval=3600):
        """
        Initialize with a key provider function
        
        Args:
            key_provider: Function that returns a fresh API key
            rotation_interval: Seconds between key rotations
        """
        self.client = SyncModelCallingClient()
        self.key_provider = key_provider
        self.rotation_interval = rotation_interval
        self.provider = None
        
        # Start rotation thread
        self.running = True
        self.rotation_thread = threading.Thread(target=self._rotation_loop)
        self.rotation_thread.daemon = True
        self.rotation_thread.start()
    
    def _rotation_loop(self):
        """Background thread that rotates keys"""
        while self.running:
            # Sleep with jitter to avoid synchronized rotations
            jitter = random.uniform(0.8, 1.2)
            time.sleep(self.rotation_interval * jitter)
            
            if self.provider:
                try:
                    # Get a fresh key
                    new_key = self.key_provider()
                    
                    # Update the registry with the new key
                    registry.update_config(
                        prefix=self.provider,
                        config={"api_key": new_key}
                    )
                    
                    print(f"Rotated API key for {self.provider}")
                except Exception as e:
                    print(f"Error rotating key: {str(e)}")
    
    def chat_completion(self, model, **kwargs):
        """Make a request, tracking the provider for rotation"""
        # Extract provider from model name
        if "/" in model:
            self.provider = model.split("/")[0]
        
        return self.client.chat_completion(model=model, **kwargs)
    
    def close(self):
        """Stop the rotation thread and close the client"""
        self.running = False
        self.client.close()
        self.rotation_thread.join(timeout=1.0)


# Example key provider function
def get_openai_key():
    """Get a fresh API key from environment or a key management system"""
    # In a real system, this might get a key from a secure vault
    return os.environ.get("OPENAI_API_KEY_CURRENT")

# Usage
client = KeyRotationClient(key_provider=get_openai_key, rotation_interval=3600)

try:
    # Use as normal
    response = client.chat_completion(
        model="openai/gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(response["choices"][0]["message"]["content"])
finally:
    client.close()
```

### Custom Authentication

Example of implementing custom authentication:

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import time

# Security scheme
security = HTTPBearer()

# Secret key for JWT
SECRET_KEY = "your-secret-key"  # In production, use a secure key

# Custom authentication middleware
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token for API access"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        
        # Check if token is expired
        if payload["exp"] < time.time():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        
        # Return the user info from token
        return payload
    
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

# Create a JWT token
def create_token(user_id, expires_in=3600):
    """Create a JWT token for API access"""
    payload = {
        "sub": str(user_id),
        "iat": int(time.time()),
        "exp": int(time.time()) + expires_in
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

# Add to FastAPI app
app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(request_data: dict, user: dict = Depends(verify_token)):
    """Protected endpoint that requires authentication"""
    # Process as normal, but now with authenticated user
    # You can also rate-limit based on user.sub
    
    # Continue with normal processing...
    pass
```

## Advanced Function Calling

### Parallel Function Calling

Example of handling multiple function calls in parallel:

```python
import asyncio
import json
from model_calling.client import ModelCallingClient

async def parallel_function_calling(query):
    """Example of processing multiple function calls in parallel"""
    client = ModelCallingClient()
    
    # Define available tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Search a database for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "Search the company knowledge base",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    try:
        # Initial request to determine what functions to call
        response = await client.chat_completion(
            model="openai/gpt-4",  # A model that supports parallel function calling
            messages=[
                {"role": "user", "content": query}
            ],
            tools=tools
        )
        
        assistant_message = response["choices"][0]["message"]
        
        # Check if parallel function calls were requested
        if "tool_calls" in assistant_message:
            tool_calls = assistant_message["tool_calls"]
            
            # Prepare function call tasks
            function_tasks = []
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                
                # Create tasks to execute functions in parallel
                if function_name == "search_database":
                    task = asyncio.create_task(search_database(arguments["query"]))
                elif function_name == "search_knowledge_base":
                    task = asyncio.create_task(search_knowledge_base(arguments["query"]))
                elif function_name == "search_web":
                    task = asyncio.create_task(search_web(arguments["query"]))
                else:
                    continue
                
                function_tasks.append((tool_call["id"], function_name, task))
            
            # Execute all functions in parallel
            function_results = []
            for tool_id, function_name, task in function_tasks:
                try:
                    result = await task
                    function_results.append({
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(result)
                    })
                except Exception as e:
                    function_results.append({
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps({"error": str(e)})
                    })
            
            # Continue the conversation with all function results
            messages = [
                {"role": "user", "content": query},
                assistant_message
            ]
            messages.extend(function_results)
            
            final_response = await client.chat_completion(
                model="openai/gpt-4",
                messages=messages
            )
            
            return final_response["choices"][0]["message"]["content"]
        else:
            return assistant_message["content"]
    
    finally:
        await client.close()

# Example function implementations
async def search_database(query):
    """Simulate searching a database"""
    await asyncio.sleep(1)  # Simulate DB lookup
    return {"results": ["Result 1", "Result 2"], "source": "database"}

async def search_knowledge_base(query):
    """Simulate searching a knowledge base"""
    await asyncio.sleep(1.5)  # Simulate KB lookup
    return {"results": ["KB Result 1", "KB Result 2"], "source": "knowledge_base"}

async def search_web(query):
    """Simulate searching the web"""
    await asyncio.sleep(2)  # Simulate web search
    return {"results": ["Web Result 1", "Web Result 2"], "source": "web"}

# Example usage
async def main():
    result = await parallel_function_calling(
        "I need information about our company's pricing strategy and any market trends."
    )
    print(result)

asyncio.run(main())
```

### Function Registry Pattern

Example of implementing a centralized function registry:

```python
import json
import inspect
from typing import Callable, Dict, Any, List

class FunctionRegistry:
    """Registry for managing and dispatching function calls"""
    
    def __init__(self):
        self.functions = {}
        self.function_schemas = {}
    
    def register(self, func: Callable, description: str = None, schema: Dict = None):
        """
        Register a function with the registry
        
        Args:
            func: The function to register
            description: Description of what the function does
            schema: Optional custom schema (otherwise generated from signature)
        """
        func_name = func.__name__
        
        # Create schema from function signature if not provided
        if schema is None:
            schema = self._generate_schema(func, description)
        
        # Store function and schema
        self.functions[func_name] = func
        self.function_schemas[func_name] = schema
        
        return self  # Allow chaining
    
    def _generate_schema(self, func: Callable, description: str = None) -> Dict:
        """Generate JSON schema from function signature"""
        sig = inspect.signature(func)
        docstring = func.__doc__ or ""
        
        # Use provided description or docstring
        desc = description or docstring.strip().split("\n")[0]
        
        # Create parameters schema
        properties = {}
        required = []
        
        for name, param in sig.parameters.items():
            # Skip self for methods
            if name == "self":
                continue
                
            # Add to required if no default value
            if param.default is inspect.Parameter.empty:
                required.append(name)
            
            # Extract annotation
            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                param_type = "string"  # default
            elif annotation == str:
                param_type = "string"
            elif annotation == int:
                param_type = "integer"
            elif annotation == float:
                param_type = "number"
            elif annotation == bool:
                param_type = "boolean"
            elif annotation == list:
                param_type = "array"
            elif annotation == dict:
                param_type = "object"
            else:
                param_type = "string"  # fallback
            
            # Extract param description from docstring if available
            param_desc = ""
            for line in docstring.split("\n"):
                if f":param {name}:" in line:
                    param_desc = line.split(f":param {name}:")[1].strip()
                    break
            
            # Create property schema
            properties[name] = {
                "type": param_type,
                "description": param_desc or f"Parameter {name}"
            }
        
        # Create the complete schema
        return {
            "name": func.__name__,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
    def get_tool_definitions(self) -> List[Dict]:
        """Get tool definitions for all registered functions"""
        tools = []
        for name, schema in self.function_schemas.items():
            tools.append({
                "type": "function",
                "function": schema
            })
        return tools
    
    async def dispatch(self, function_call: Dict) -> Any:
        """Dispatch a function call to the appropriate registered function"""
        name = function_call.get("name")
        args_str = function_call.get("arguments", "{}")
        
        if name not in self.functions:
            raise ValueError(f"Function {name} not registered")
        
        # Parse arguments
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid arguments: {args_str}")
        
        # Call the function
        func = self.functions[name]
        try:
            if inspect.iscoroutinefunction(func):
                result = await func(**args)
            else:
                result = func(**args)
            return result
        except Exception as e:
            return {"error": str(e), "type": type(e).__name__}


# Example usage
registry = FunctionRegistry()

# Register some functions
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get current weather for a location
    
    :param location: City name or coordinates
    :param unit: Temperature unit (celsius or fahrenheit)
    """
    # Implementation...
    return {"temperature": 22, "unit": unit, "location": location}

def search_database(query: str, limit: int = 10) -> list:
    """Search the database for information
    
    :param query: Search query
    :param limit: Maximum number of results
    """
    # Implementation...
    return [f"Result {i}" for i in range(limit)]

async def fetch_user_profile(user_id: int) -> dict:
    """Fetch a user profile from the database
    
    :param user_id: User ID
    """
    # Async implementation...
    return {"id": user_id, "name": "Example User"}

# Register functions
registry.register(get_weather)
registry.register(search_database)
registry.register(fetch_user_profile)

# Get tool definitions
tools = registry.get_tool_definitions()

# Example dispatch
async def handle_llm_response(function_call):
    """Handle a function call from an LLM"""
    result = await registry.dispatch(function_call)
    return result

# Example function call from LLM
function_call = {
    "name": "get_weather",
    "arguments": '{"location": "San Francisco", "unit": "fahrenheit"}'
}

import asyncio
result = asyncio.run(handle_llm_response(function_call))
print(result)
```

## Working with Model Preferences

### Model Selection Strategy

Example of implementing a model selection strategy:

```python
class ModelSelector:
    """Strategy for selecting the best model for a task"""
    
    def __init__(self):
        """Initialize with model capabilities and costs"""
        self.models = {
            # Local models
            "ollama/mistral-small3.1:24b": {
                "capabilities": {
                    "function_calling": True,
                    "code_generation": 0.7,
                    "creative_writing": 0.8,
                    "factual_accuracy": 0.7
                },
                "cost": 0,
                "latency": 1.5,  # relative latency
                "context_length": 8192
            },
            "ollama/llama3:8b": {
                "capabilities": {
                    "function_calling": True,
                    "code_generation": 0.6,
                    "creative_writing": 0.7,
                    "factual_accuracy": 0.6
                },
                "cost": 0,
                "latency": 1.0,
                "context_length": 4096
            },
            
            # Hosted models
            "openai/gpt-4": {
                "capabilities": {
                    "function_calling": True,
                    "code_generation": 0.95,
                    "creative_writing": 0.9,
                    "factual_accuracy": 0.9
                },
                "cost": 0.03,  # $ per 1K tokens
                "latency": 2.5,
                "context_length": 8192
            },
            "anthropic/claude-3-opus-20240229": {
                "capabilities": {
                    "function_calling": True,
                    "code_generation": 0.9,
                    "creative_writing": 0.95,
                    "factual_accuracy": 0.9
                },
                "cost": 0.025,
                "latency": 2.0,
                "context_length": 100000
            }
        }
    
    def select_for_task(self, task_type, 
                        budget=None, 
                        latency_preference="medium", 
                        context_length=None):
        """
        Select the best model for a specific task
        
        Args:
            task_type: Type of task (code_generation, creative_writing, etc.)
            budget: Maximum cost per 1K tokens (None for unlimited)
            latency_preference: "low", "medium", or "high"
            context_length: Minimum required context length
        
        Returns:
            Best model name for the task
        """
        # Filter by budget
        candidates = self.models.copy()
        if budget is not None:
            candidates = {
                name: info for name, info in candidates.items()
                if info["cost"] <= budget
            }
        
        # Filter by context length
        if context_length is not None:
            candidates = {
                name: info for name, info in candidates.items()
                if info["context_length"] >= context_length
            }
        
        # Filter by capability
        if not candidates:
            return None
            
        # Convert latency preference to numeric
        latency_scores = {"low": 3.0, "medium": 2.0, "high": 1.0}
        latency_weight = latency_scores.get(latency_preference, 2.0)
        
        # Score each model for this task
        scores = {}
        for name, info in candidates.items():
            capabilities = info["capabilities"]
            
            # Base score from capability for this task
            if task_type in capabilities:
                base_score = capabilities[task_type]
            else:
                # Average of all capabilities as fallback
                base_score = sum(capabilities.values()) / len(capabilities)
            
            # Adjust for latency (weighted inverse)
            latency_factor = latency_weight / info["latency"]
            
            # Final score - could add more factors here
            scores[name] = base_score * latency_factor
        
        # Return best model
        if not scores:
            return None
            
        return max(scores, key=scores.get)

# Example usage
selector = ModelSelector()

# Get model recommendations for different tasks
code_model = selector.select_for_task("code_generation", budget=0.01)
creative_model = selector.select_for_task("creative_writing", latency_preference="high")
long_context_model = selector.select_for_task("factual_accuracy", context_length=50000)

print(f"Recommended for coding: {code_model}")
print(f"Recommended for creative tasks: {creative_model}")
print(f"Recommended for long context: {long_context_model}")
```

## Advanced Topics Conclusion

These advanced techniques allow you to get the most out of the model-calling service. By customizing adapters, optimizing performance, and implementing security measures, you can create robust and efficient LLM applications that work with multiple models and providers.
