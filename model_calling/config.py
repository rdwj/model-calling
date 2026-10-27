"""
Configuration module for model-calling service.
"""
import os
from typing import Dict, Any

# Default configurations
DEFAULT_CONFIG = {
    "ollama": {
        "base_url": "http://localhost:11434",
        # Function-call capable models
        "function_call_models": [
            "mistral", "mistral-small", "mistral-large",
            "llama3", "llama-3", 
            "neural-chat", "openhermes", "nous-hermes", 
            "qwen", "yi", "gemma"
        ]
    },
    "vllm": {
        "base_url": "http://localhost:8000",
        "openai_compatible": True,
        # Add any vLLM specific settings here
    }
}

# Override with environment variables
def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables and defaults"""
    config = DEFAULT_CONFIG.copy()
    
    # Ollama settings
    if os.environ.get("OLLAMA_BASE_URL"):
        config["ollama"]["base_url"] = os.environ.get("OLLAMA_BASE_URL")
    
    # vLLM settings
    if os.environ.get("VLLM_BASE_URL"):
        config["vllm"]["base_url"] = os.environ.get("VLLM_BASE_URL")
    
    if os.environ.get("VLLM_OPENAI_COMPATIBLE"):
        config["vllm"]["openai_compatible"] = os.environ.get("VLLM_OPENAI_COMPATIBLE").lower() == "true"
    
    return config


# Current active configuration
active_config = load_config()


def get_config(provider: str = None) -> Dict[str, Any]:
    """Get configuration, optionally for a specific provider"""
    if provider:
        return active_config.get(provider, {})
    return active_config


def update_config(provider: str, updates: Dict[str, Any]) -> None:
    """Update configuration for a provider"""
    if provider in active_config:
        active_config[provider].update(updates)
    else:
        active_config[provider] = updates
