"""
Cohere API adapter
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional
import httpx
import time

from model_calling.hosted import HostedProviderAdapter
from model_calling.adapter import create_openai_response, create_openai_streaming_chunk

logger = logging.getLogger(__name__)


class CohereAdapter(HostedProviderAdapter):
    """Adapter for Cohere API"""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize Cohere adapter"""
        model_config["provider_name"] = "cohere"
        super().__init__(model_config)
    
    def _get_api_key(self) -> str:
        """Get Cohere API key from environment"""
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            logger.warning("COHERE_API_KEY not found in environment")
        return api_key
    
    def _get_base_url(self) -> str:
        """Get Cohere API base URL"""
        return os.environ.get("COHERE_API_BASE", "https://api.cohere.ai/v1/chat")
    
    async def _make_request_headers(self) -> Dict[str, str]:
        """Create headers for the Cohere API request"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    async def _call_model_api(self, translated_request: Dict[str, Any]) -> Dict[str, Any]:
        """Call the Cohere API and return the raw response"""
        headers = await self._make_request_headers()
        is_streaming = translated_request.get("stream", False)
        async with httpx.AsyncClient() as client:
            if is_streaming:
                async def stream_request():
                    url = f"{self._get_base_url()}/stream"
                    async with client.post(url, headers=headers, json=translated_request, stream=True) as response:
                        async for chunk in response.aiter_bytes():
                            yield chunk
                return stream_request()
            else:
                url = f"{self._get_base_url()}"
                response = await client.post(url, headers=headers, json=translated_request)
                return response.json()
    
    async def translate_request(self, openai_format_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI format to Cohere format"""
        messages = openai_format_request.get("messages", [])
        
        # Extract system message if present
        system = ""
        chat_history = []
        
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_history.append(msg)
                
        # The last user message becomes the actual query
        query = ""
        if chat_history and chat_history[-1]["role"] == "user":
            query = chat_history[-1]["content"]
            chat_history = chat_history[:-1]
            
        # Convert to Cohere format
        cohere_request = {
            "model": self.model_name,
            "query": query,
            "stream": openai_format_request.get("stream", False)
        }
        
        # Add preamble (system message) if present
        if system:
            cohere_request["preamble"] = system
            
        # Add temperature if present
        if "temperature" in openai_format_request:
            cohere_request["temperature"] = openai_format_request["temperature"]
        
        # Convert chat history
        if chat_history:
            cohere_history = []
            for msg in chat_history:
                role = "USER" if msg["role"] == "user" else "CHATBOT"
                cohere_history.append({
                    "role": role,
                    "message": msg["content"]
                })
            
            cohere_request["chat_history"] = cohere_history
            
        # Handle tools (if supported in the future)
        if "tools" in openai_format_request and await self.supports_tools():
            tools = []
            for tool in openai_format_request["tools"]:
                if tool["type"] == "function":
                    # Convert to Cohere format if/when they add function calling
                    pass
            
            # Add tools when Cohere supports them
            # cohere_request["tools"] = tools
            
        return cohere_request
    
    async def translate_response(self, cohere_response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Cohere response to OpenAI format"""
        if "error" in cohere_response:
            return cohere_response
            
        # Extract the response text
        text = cohere_response.get("text", "")
            
        # Create OpenAI-compatible response
        return create_openai_response(
            model=f"cohere/{self.model_name}",
            content=text,
            finish_reason="stop"
        )
    
    async def translate_streaming_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Translate a Cohere streaming chunk to OpenAI format"""
        # Extract content from the chunk
        text = chunk.get("text", "")
        is_finished = chunk.get("is_finished", False)
        
        # Convert to OpenAI streaming format
        return create_openai_streaming_chunk(
            model=f"cohere/{self.model_name}",
            content=text,
            is_final=is_finished
        )
    
    async def supports_tools(self) -> bool:
        """Check if this Cohere model supports function calling"""
        # As of now, Cohere doesn't support function calling
        # This can be updated when they add support
        return False


# Available Cohere models
COHERE_MODELS = {
    "command": {
        "supports_function_call": False,
        "supports_vision": False
    },
    "command-light": {
        "supports_function_call": False,
        "supports_vision": False
    },
    "command-r": {
        "supports_function_call": False,
        "supports_vision": False
    },
    "command-r-plus": {
        "supports_function_call": False,
        "supports_vision": False
    }
}
