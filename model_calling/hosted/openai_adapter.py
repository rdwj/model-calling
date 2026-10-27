"""
OpenAI API adapter
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional, Generator
import httpx
import time

from model_calling.hosted import HostedProviderAdapter
from model_calling.adapter import create_openai_response, create_openai_streaming_chunk

logger = logging.getLogger(__name__)


class OpenAIAdapter(HostedProviderAdapter):
    """Adapter for OpenAI API"""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize OpenAI adapter"""
        model_config["provider_name"] = "openai"
        super().__init__(model_config)
        self.org_id = model_config.get("org_id") or os.environ.get("OPENAI_ORG_ID")
    
    def _get_api_key(self) -> str:
        """Get OpenAI API key from environment"""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment")
        return api_key
    
    def _get_base_url(self) -> str:
        """Get OpenAI API base URL"""
        return os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1/chat/completions")
    
    async def _make_request_headers(self) -> Dict[str, str]:
        """Create headers for the OpenAI API request"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        if self.org_id:
            headers["OpenAI-Organization"] = self.org_id
        
        return headers
    
    async def translate_request(self, openai_format_request: Dict[str, Any]) -> Dict[str, Any]:
        """Format is already OpenAI-compatible, just update model name"""
        translated_request = openai_format_request.copy()
        translated_request["model"] = self.model_name
        return translated_request
    
    async def translate_response(self, openai_response: Dict[str, Any]) -> Dict[str, Any]:
        """Response is already in OpenAI format, just update model name if needed"""
        if "error" in openai_response:
            return openai_response
            
        if "model" in openai_response:
            openai_response["model"] = f"openai/{self.model_name}"
        
        return openai_response
    
    async def translate_streaming_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Translate an OpenAI streaming chunk"""
        if "model" in chunk:
            chunk["model"] = f"openai/{self.model_name}"
        
        return chunk
    
    async def supports_tools(self) -> bool:
        """Check if this OpenAI model supports function calling"""
        # List of OpenAI models that support function/tool calling
        function_call_models = [
            "gpt-4", "gpt-4-turbo", "gpt-4-0125", "gpt-4-1106", "gpt-4-vision",
            "gpt-3.5-turbo", "gpt-3.5-turbo-1106"
        ]
        
        # Check if the model name starts with any of these prefixes
        return any(self.model_name.startswith(model) for model in function_call_models)


# List of available OpenAI models and their capabilities
OPENAI_MODELS = {
    "gpt-4": {
        "supports_function_call": True,
        "supports_vision": False
    },
    "gpt-4-turbo": {
        "supports_function_call": True,
        "supports_vision": False
    },
    "gpt-4-0125": {
        "supports_function_call": True,
        "supports_vision": False
    },
    "gpt-4-1106": {
        "supports_function_call": True, 
        "supports_vision": False
    },
    "gpt-4-vision-preview": {
        "supports_function_call": True,
        "supports_vision": True
    },
    "gpt-3.5-turbo": {
        "supports_function_call": True,
        "supports_vision": False
    },
    "gpt-3.5-turbo-1106": {
        "supports_function_call": True,
        "supports_vision": False
    }
}
