"""
Model registry for managing and selecting appropriate adapters.
"""

import logging
import os
import httpx
import json
import asyncio
import time
from typing import Dict, Any, Optional, Type, List, Set, Tuple

from model_calling.adapter import ModelAdapter
from model_calling.ollama_adapter import OllamaAdapter
from model_calling.thinking import ThinkingStyle
from model_calling.callbacks import DefaultCallback
from model_calling.library import Client
from model_calling.capability_cache import capability_cache
from model_calling.config import backend_overrides

logger = logging.getLogger(__name__)

# Import the capability testing module
try:
    from model_calling.capability_testing import CapabilityTester

    capability_tester = CapabilityTester(base_url="http://localhost:11434")
except ImportError:
    logger.warning("CapabilityTester not available. Falling back to basic capability detection.")
    capability_tester = None

# Import embedding adapters
from model_calling.embedding_adapter import EmbeddingAdapter
from model_calling.hosted.cohere_embedding_adapter import CohereEmbeddingAdapter

# Import hosted providers
from model_calling.hosted.openai_adapter import OpenAIAdapter, OPENAI_MODELS
from model_calling.hosted.cohere_adapter import CohereAdapter, COHERE_MODELS


class ModelCapabilities:
    """Class to store and manage model capabilities"""

    def __init__(self):
        self.supports_function_call = False
        self.supports_streaming = False
        self.supports_system_messages = False
        self.supports_vision = False
        self.supports_json_mode = False
        self.supports_reasoning = False
        self.thinking_style = ThinkingStyle.NONE
        self.max_tokens = None
        self.verified = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert capabilities to dictionary"""
        return {
            "supports_function_call": self.supports_function_call,
            "supports_streaming": self.supports_streaming,
            "supports_system_messages": self.supports_system_messages,
            "supports_vision": self.supports_vision,
            "supports_json_mode": self.supports_json_mode,
            "supports_reasoning": self.supports_reasoning,
            "thinking_style": self.thinking_style.value,
            "max_tokens": self.max_tokens,
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCapabilities":
        """Create capabilities from dictionary"""
        capabilities = cls()
        capabilities.supports_function_call = data.get("supports_function_call", False)
        capabilities.supports_streaming = data.get("supports_streaming", False)
        capabilities.supports_system_messages = data.get("supports_system_messages", False)
        capabilities.supports_vision = data.get("supports_vision", False)
        capabilities.supports_json_mode = data.get("supports_json_mode", False)
        capabilities.supports_reasoning = data.get("supports_reasoning", False)
        capabilities.supports_thinking = data.get("supports_thinking", False)
        capabilities.max_tokens = data.get("max_tokens")
        capabilities.verified = data.get("verified", False)
        return capabilities


class ModelRegistry:
    """Registry for model adapters"""

    def __init__(self):
        """Initialize the registry with default configurations"""
        self.adapters = {}
        self.embedding_adapters = {}
        self.model_capabilities = {}
        self.adapter_classes = {
            "ollama": OllamaAdapter,
            "openai": OpenAIAdapter,
            "cohere": CohereAdapter,
        }

        # Embedding adapters for the prefixes `client` doesn't serve yet.
        self.embedding_adapter_classes = {
            "cohere": CohereEmbeddingAdapter,
        }

        # Create a default callback for logging
        self.default_callback = DefaultCallback()
        # The library client the server answers ported prefixes with; closed
        # when the server stops. Unsupported parameters are dropped, as the
        # gateway always has. Backend URLs and keys set in the environment
        # override the library defaults.
        self.client = Client(
            config=backend_overrides(), callbacks=[self.default_callback], drop_params=True
        )

        # Default configurations. The ported prefixes' entries are the client's
        # own config dicts, so an update here reaches the next request.
        self.default_configs = {
            **self.client.config,
            "cohere": {
                "base_url": "https://api.cohere.ai/v1/chat",
                "api_key": os.environ.get("COHERE_API_KEY", ""),
                "callbacks": [self.default_callback],
            },
        }

    def register_adapter_class(self, prefix: str, adapter_class: Type[ModelAdapter]):
        """Register a new adapter class for a specific model prefix"""
        self.adapter_classes[prefix] = adapter_class

    def register_embedding_adapter_class(self, prefix: str, adapter_class: Type[EmbeddingAdapter]):
        """Register a new embedding adapter class for a specific model prefix"""
        self.embedding_adapter_classes[prefix] = adapter_class

    async def detect_ollama_capabilities(
        self, model_name: str, force_retest: bool = False
    ) -> ModelCapabilities:
        """Detect capabilities of an Ollama model through comprehensive testing

        Args:
            model_name: The name of the model to test
            force_retest: If True, ignore cache and force fresh testing

        Returns:
            ModelCapabilities: The detected capabilities
        """
        # Check the cache first (unless force_retest is True)
        cache_key = f"ollama/{model_name}"
        cached_data = None

        if not force_retest:
            cached_data = capability_cache.get_capability(cache_key)

        if not force_retest and cached_data and capability_cache.is_cache_valid(cache_key):
            logger.info(f"Using cached capabilities for {model_name}")
            capabilities = ModelCapabilities.from_dict(cached_data)
            return capabilities

        logger.info(f"Testing all capabilities for {model_name}...")

        # Create a new capabilities object
        capabilities = ModelCapabilities()
        base_url = self.default_configs["ollama"]["base_url"]

        # Use the capability tester if available
        if capability_tester is not None:
            try:
                # Run comprehensive capability testing
                logger.info(f"Running comprehensive capability tests for {model_name}...")
                # Probe the configured Ollama, which env or config updates may move.
                capability_tester.base_url = base_url
                test_results = await capability_tester.test_all_capabilities(model_name)

                # Update capabilities based on test results
                capabilities.supports_function_call = test_results.get(
                    "supports_function_call", False
                )
                capabilities.supports_streaming = test_results.get("supports_streaming", True)
                capabilities.supports_system_messages = test_results.get(
                    "supports_system_messages", True
                )
                capabilities.supports_vision = test_results.get("supports_vision", False)
                capabilities.supports_json_mode = test_results.get("supports_json_mode", False)
                capabilities.supports_reasoning = test_results.get("supports_reasoning", False)

                # Determine thinking style
                if test_results.get("supports_thinking", False):
                    if "granite" in model_name.lower():
                        capabilities.thinking_style = ThinkingStyle.GRANITE
                    else:
                        capabilities.thinking_style = ThinkingStyle.GENERAL
                else:
                    capabilities.thinking_style = ThinkingStyle.NONE

                # Mark as verified
                capabilities.verified = True

                # Cache the results
                cache_data = capabilities.to_dict()
                cache_data["last_tested"] = test_results.get("last_tested", int(time.time()))
                capability_cache.set_capability(cache_key, cache_data)

                # Log detailed results
                logger.info(
                    f"Capability testing complete for {model_name}: {capabilities.to_dict()}"
                )

                return capabilities

            except Exception as e:
                logger.warning(f"Capability testing failed for {model_name}: {e}")
                # Fall through to basic detection

        # Fallback: Use basic capability detection if comprehensive testing failed or tester not available
        try:
            # First check if the model exists
            async with httpx.AsyncClient() as client:
                logger.info(f"Verifying model {model_name} exists...")
                response = await client.post(
                    f"{base_url}/api/show", json={"name": model_name}, timeout=10.0
                )

                if response.status_code != 200:
                    logger.warning(f"Model {model_name} not found: {response.status_code}")
                    return capabilities

                # Get basic model info for logging
                model_info = response.json()
                model_family = model_info.get("details", {}).get("family", "").lower()
                model_parameter_count = model_info.get("details", {}).get("parameter_count", 0)
                model_quantization = model_info.get("details", {}).get("quantization_level", "")

                logger.info(
                    f"Model info: {model_name} ({model_family}, {model_parameter_count} params, {model_quantization})"
                )

                # Set defaults for basic capabilities
                capabilities.supports_streaming = True
                capabilities.supports_system_messages = True

                # Test function calling capability using the adapter's method
                logger.info(f"Testing function calling capability for {model_name}...")
                config = self.default_configs["ollama"].copy()
                config["model_name"] = model_name
                config["supports_function_call"] = True  # Enable for testing

                test_adapter = OllamaAdapter(config)
                supports_tools, tool_test_log = await test_adapter.detect_tool_call_support()

                capabilities.supports_function_call = supports_tools
                logger.info(f"Function calling support for {model_name}: {supports_tools}")

                # Set vision based on model name (fallback)
                if any(
                    vision_model in model_name.lower()
                    for vision_model in ["llava", "bakllava", "vision"]
                ):
                    capabilities.supports_vision = True

                # Mark as verified
                capabilities.verified = True

                # Cache the results
                capability_cache.set_capability(cache_key, capabilities.to_dict())

                return capabilities

        except Exception as e:
            logger.warning(f"Failed to detect capabilities for {model_name}: {e}")

            # Set safe defaults
            capabilities.supports_streaming = True
            capabilities.supports_system_messages = True

            # Cache even these basic capabilities
            capabilities.verified = False
            capability_cache.set_capability(cache_key, capabilities.to_dict())

            return capabilities

    def get_capabilities(self, model_name: str, ignore_cache: bool = False) -> ModelCapabilities:
        """Get capabilities for a model

        Args:
            model_name: The model name to get capabilities for
            ignore_cache: If True, don't use in-memory cache

        Returns:
            ModelCapabilities: The capabilities for the model
        """
        # Check if we already have capabilities in memory and we're not ignoring cache
        if not ignore_cache and model_name in self.model_capabilities:
            return self.model_capabilities[model_name]

        # Check if we have cached capabilities
        cache_key = model_name
        cached_data = capability_cache.get_capability(cache_key)
        if cached_data:
            logger.info(f"Using cached capabilities for {model_name}")
            capabilities = ModelCapabilities.from_dict(cached_data)
            self.model_capabilities[model_name] = capabilities
            return capabilities

        # Create default capabilities
        capabilities = ModelCapabilities()

        # Parse the model prefix
        parts = model_name.split("/", 1)
        if len(parts) == 1:
            prefix = "ollama"
            short_name = model_name
        else:
            prefix = parts[0]
            short_name = parts[1]

        # Set capabilities based on model type
        if prefix == "ollama":
            # We no longer assume function call support based on name
            # This will be tested and set during verification
            capabilities.supports_function_call = False
            capabilities.supports_streaming = True
            capabilities.supports_system_messages = True

            # Set JSON and reasoning capabilities based on model name
            # These are less critical than function calling, so we can still use name heuristics
            if any(
                json_model in short_name.lower()
                for json_model in ["gpt", "mistral", "llama3", "claude", "gemma", "mixtral"]
            ):
                capabilities.supports_json_mode = True

            if any(
                reasoning_model in short_name.lower()
                for reasoning_model in [
                    "gpt",
                    "mistral",
                    "llama3",
                    "claude",
                    "gemma",
                    "mixtral",
                    "qwen",
                    "phi",
                ]
            ):
                capabilities.supports_reasoning = True
        elif prefix == "openai":
            for model_prefix, model_caps in OPENAI_MODELS.items():
                if short_name.startswith(model_prefix):
                    capabilities.supports_function_call = model_caps.get(
                        "supports_function_call", False
                    )
                    capabilities.supports_streaming = True
                    capabilities.supports_system_messages = True
                    capabilities.supports_vision = model_caps.get("supports_vision", False)
                    capabilities.verified = True
                    break
        elif prefix == "cohere":
            for model_prefix, model_caps in COHERE_MODELS.items():
                if short_name.startswith(model_prefix):
                    capabilities.supports_function_call = model_caps.get(
                        "supports_function_call", False
                    )
                    capabilities.supports_streaming = True
                    capabilities.supports_system_messages = True
                    capabilities.verified = True
                    break

        # Store capabilities
        self.model_capabilities[model_name] = capabilities
        return capabilities

    async def verify_capabilities(
        self, model_name: str, force_retest: bool = False
    ) -> ModelCapabilities:
        """Verify and update capabilities for a model

        Args:
            model_name: The model name or ID to verify
            force_retest: If True, ignore cache and force a fresh test

        Returns:
            ModelCapabilities: The verified capabilities
        """
        start_time = time.time()
        logger.info(f"Verifying capabilities for {model_name}...")

        # Parse the model prefix
        parts = model_name.split("/", 1)
        if len(parts) == 1:
            prefix = "ollama"
            short_name = model_name
        else:
            prefix = parts[0]
            short_name = parts[1]

        # Check if capabilities are cached and we're not forcing a retest
        if not force_retest:
            # First try the exact model name
            cache_key = model_name
            cached_data = capability_cache.get_capability(cache_key)
            if cached_data and capability_cache.is_cache_valid(cache_key):
                logger.info(f"Using cached verified capabilities for {model_name}")
                capabilities = ModelCapabilities.from_dict(cached_data)
                self.model_capabilities[model_name] = capabilities
                return capabilities

            # Also check provider/model format
            provider_key = f"{prefix}/{short_name}"
            cached_data = capability_cache.get_capability(provider_key)
            if cached_data and capability_cache.is_cache_valid(provider_key):
                logger.info(f"Using cached verified capabilities from {provider_key}")
                capabilities = ModelCapabilities.from_dict(cached_data)
                self.model_capabilities[model_name] = capabilities
                return capabilities

        # If forcing retest or no valid cache, run actual capability detection
        logger.info(
            f"Running comprehensive capability detection for {model_name} (force_retest={force_retest})"
        )

        # Clear all related cache entries
        if force_retest:
            cache_keys = [model_name, f"{prefix}/{short_name}", short_name]
            for key in cache_keys:
                capability_cache.set_capability(key, {})
                # Also remove from in-memory cache
                if key in self.model_capabilities:
                    del self.model_capabilities[key]
            logger.info(f"Cleared all capability caches for {model_name}")

        # Verify capabilities based on provider
        capabilities = None
        if prefix == "ollama":
            capabilities = await self.detect_ollama_capabilities(
                short_name, force_retest=force_retest
            )
        elif prefix == "vllm":
            capabilities = await self.detect_vllm_capabilities(short_name)
        elif prefix == "openai":
            # For cloud services, we use predefined capabilities
            capabilities = self.get_capabilities(model_name, ignore_cache=force_retest)
        elif prefix == "anthropic":
            capabilities = self.get_capabilities(model_name, ignore_cache=force_retest)
        elif prefix == "cohere":
            capabilities = self.get_capabilities(model_name, ignore_cache=force_retest)
        else:
            logger.warning(f"Unknown provider prefix: {prefix}")
            capabilities = ModelCapabilities()

        # Store the capabilities in our in-memory cache
        self.model_capabilities[model_name] = capabilities

        # Also store under the provider/model format
        if prefix != "ollama" or model_name != f"ollama/{short_name}":
            self.model_capabilities[f"{prefix}/{short_name}"] = capabilities

        elapsed_time = time.time() - start_time
        logger.info(
            f"Capability verification for {model_name} completed in {elapsed_time:.2f} seconds"
        )
        return capabilities

    def get_adapter(self, model_name: str) -> Optional[ModelAdapter]:
        """Get or create an adapter for the specified model"""
        # Check if we already have this adapter
        if model_name in self.adapters:
            return self.adapters[model_name]

        # Parse the model prefix to determine which adapter to use
        parts = model_name.split("/", 1)
        if len(parts) == 1:
            # No prefix, assume it's ollama by default
            prefix = "ollama"
            short_name = model_name
        else:
            prefix = parts[0]
            short_name = parts[1]

        # Get the adapter class
        adapter_class = self.adapter_classes.get(prefix)
        if not adapter_class:
            logger.error(f"No adapter found for prefix: {prefix}")
            return None

        # Get default config
        config = self.default_configs.get(prefix, {}).copy()
        config["model_name"] = short_name

        # Check cache for capabilities first
        cache_key = model_name
        cached_data = capability_cache.get_capability(cache_key)

        if cached_data and capability_cache.is_cache_valid(cache_key):
            logger.info(f"Using cached capabilities for adapter creation: {model_name}")
            capabilities = ModelCapabilities.from_dict(cached_data)
            # Update in-memory cache
            self.model_capabilities[model_name] = capabilities
        else:
            # Get capabilities from in-memory cache
            capabilities = self.get_capabilities(model_name)

        # Add capabilities to config
        config["supports_function_call"] = capabilities.supports_function_call
        config["supports_streaming"] = capabilities.supports_streaming
        config["supports_system_messages"] = capabilities.supports_system_messages
        config["supports_vision"] = capabilities.supports_vision
        config["supports_json_mode"] = capabilities.supports_json_mode
        config["supports_reasoning"] = capabilities.supports_reasoning
        config["capabilities_verified"] = capabilities.verified

        # Create and cache the adapter
        adapter = adapter_class(config)
        self.adapters[model_name] = adapter

        # Schedule capability verification if not verified or cache is stale
        if (
            not capabilities.verified
            or not cached_data
            or not capability_cache.is_cache_valid(cache_key)
        ):
            # Use a distinctive name for the task
            task_name = f"verify_{model_name.replace('/', '_')}"
            logger.info(f"Scheduling capability verification for {model_name} (task: {task_name})")
            verification_task = asyncio.create_task(self.verify_capabilities(model_name))
            verification_task.set_name(task_name)

        return adapter

    def get_embedding_adapter(self, model_name: str) -> Optional[EmbeddingAdapter]:
        """Get or create an adapter for the specified embedding model"""
        # Check if we already have this adapter
        if model_name in self.embedding_adapters:
            return self.embedding_adapters[model_name]

        # Parse the model prefix to determine which adapter to use
        parts = model_name.split("/", 1)
        if len(parts) == 1:
            # No prefix, assume it's ollama by default
            prefix = "ollama"
            short_name = model_name
        else:
            prefix = parts[0]
            short_name = parts[1]

        # Get the adapter class
        adapter_class = self.embedding_adapter_classes.get(prefix)
        if not adapter_class:
            logger.error(f"No embedding adapter found for prefix: {prefix}")
            return None

        # Get default config
        config = self.default_configs.get(prefix, {}).copy()
        config["model_name"] = short_name

        # Special case for known embedding models
        if short_name == "nomic-embed-text" or "embed" in short_name.lower():
            # Set known dimensions if available
            if "nomic-embed-text" in short_name:
                config["dimensions"] = 768
            # Else use default dimensions from adapter

        # Create and cache the adapter
        adapter = adapter_class(config)
        self.embedding_adapters[model_name] = adapter

        return adapter

    def update_config(self, prefix: str, config_updates: Dict[str, Any]):
        """Update the default configuration for a model prefix"""
        if prefix in self.default_configs:
            self.default_configs[prefix].update(config_updates)
        else:
            self.default_configs[prefix] = config_updates

        # Invalidate cached adapters for this prefix
        for model_name in list(self.adapters.keys()):
            if model_name.startswith(f"{prefix}/") or (
                prefix == "ollama" and "/" not in model_name
            ):
                del self.adapters[model_name]
                # Also remove capabilities
                if model_name in self.model_capabilities:
                    del self.model_capabilities[model_name]

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List available models from all adapters"""
        models = []

        # Add local models first
        try:
            # List Ollama models
            ollama_url = self.default_configs["ollama"]["base_url"]
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{ollama_url}/api/tags", timeout=10.0)
                if response.status_code == 200:
                    for model in response.json().get("models", []):
                        model_name = model.get("name")
                        model_entry = {"id": f"ollama/{model_name}", "object": "model"}

                        # Add capability data if available in cache
                        cache_key = f"ollama/{model_name}"
                        cached_data = capability_cache.get_capability(cache_key)
                        if cached_data:
                            model_entry["capabilities"] = {
                                "function_calling": cached_data.get(
                                    "supports_function_call", False
                                ),
                                "vision": cached_data.get("supports_vision", False),
                                "verified": cached_data.get("verified", False),
                            }
                            # Add when it was last tested
                            if "last_tested" in cached_data:
                                model_entry["last_tested"] = cached_data["last_tested"]

                        models.append(model_entry)

                        # Pre-cache any missing capabilities for quicker access later
                        if not cached_data or not capability_cache.is_cache_valid(cache_key):
                            # We don't want to block the model listing, so just schedule the task
                            task_name = f"pretest_{model_name.replace('/', '_')}"
                            verification_task = asyncio.create_task(
                                self.verify_capabilities(f"ollama/{model_name}")
                            )
                            verification_task.set_name(task_name)
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")

        try:
            # List vLLM models
            vllm_config = self.default_configs["vllm"]
            vllm_url = vllm_config["base_url"]
            api_key = vllm_config.get("api_key")
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{vllm_url}/models", headers=headers, timeout=10.0)
                if response.status_code == 200:
                    for model in response.json().get("data", []):
                        model_id = model.get("id")
                        models.append({"id": f"vllm/{model_id}", "object": "model"})
        except Exception as e:
            logger.warning(f"Failed to list vLLM models: {e}")

        # Add OpenAI models
        models.extend(
            [
                {"id": "openai/gpt-4o", "object": "model"},
                {"id": "openai/gpt-4-turbo", "object": "model"},
                {"id": "openai/gpt-4", "object": "model"},
                {"id": "openai/gpt-3.5-turbo", "object": "model"},
            ]
        )

        # Add Cohere models
        models.extend(
            [
                {"id": "cohere/command", "object": "model"},
                {"id": "cohere/command-r", "object": "model"},
                {"id": "cohere/command-r-plus", "object": "model"},
            ]
        )

        return models


# Singleton instance
registry = ModelRegistry()
