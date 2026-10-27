"""
Base module for hosted LLM provider adapters.
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import httpx

from model_calling.adapter import ModelAdapter, StreamingAdapter

logger = logging.getLogger(__name__)


class HostedProviderAdapter(ModelAdapter, StreamingAdapter):
    """Base class for hosted LLM provider adapters"""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize with model configuration"""
        super().__init__(model_config)
        self.provider_name = model_config.get("provider_name", "hosted")
        self.api_key = model_config.get("api_key") or self._get_api_key()
        self.model_name = model_config.get("model_name", "")
        
        # Load additional configuration
        self.base_url = model_config.get("base_url") or self._get_base_url()
        self.timeout = model_config.get("timeout", 60.0)
        self.max_retries = model_config.get("max_retries", 3)
    
    @abstractmethod
    def _get_api_key(self) -> str:
        """Get API key from environment or configuration"""
        pass
    
    @abstractmethod
    def _get_base_url(self) -> str:
        """Get base URL for the API"""
        pass
    
    @abstractmethod
    async def _make_request_headers(self) -> Dict[str, str]:
        """Create headers for the API request"""
        pass
    
    async def _call_model_api(self, translated_request: Dict[str, Any]) -> Dict[str, Any]:
        """Call the hosted API and return the raw response"""
        headers = await self._make_request_headers()
        is_streaming = translated_request.get("stream", False)
        
        try:
            async with httpx.AsyncClient() as client:
                if is_streaming:
                    # Handle streaming responses
                    all_chunks = []
                    messages = translated_request.get("messages", [])
                    
                    async with client.stream("POST", self.base_url, json=translated_request, headers=headers, timeout=self.timeout) as response:
                        response.raise_for_status()
                        async for chunk in response.aiter_lines():
                            if not chunk.strip():
                                continue
                            try:
                                # OpenAI format uses "data: " prefix for SSE
                                if chunk.startswith("data: "):
                                    chunk = chunk[6:]  # Remove "data: " prefix
                                
                                if chunk == "[DONE]":
                                    break
                                    
                                chunk_data = json.loads(chunk)
                                # Handle callbacks and translation
                                translated_chunk = await self.handle_streaming_chunk(
                                    chunk_data, messages, **translated_request
                                )
                                all_chunks.append(chunk_data)
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse {self.provider_name} chunk: {chunk}")
                    
                    # Return the last chunk for completion
                    return all_chunks[-1] if all_chunks else {}
                else:
                    # For non-streaming responses
                    response = await client.post(
                        self.base_url,
                        json=translated_request,
                        headers=headers,
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = {}
            try:
                error_detail = e.response.json()
            except Exception:
                error_detail["text"] = e.response.text
            
            logger.error(f"API error from {self.provider_name}: {error_detail}")
            # Return a standardized error format
            return {
                "error": {
                    "message": f"Error from {self.provider_name} API: {e.response.status_code}",
                    "type": "api_error",
                    "param": None,
                    "code": e.response.status_code,
                    "details": error_detail
                }
            }
        except Exception as e:
            logger.exception(f"Unexpected error calling {self.provider_name} API")
            return {
                "error": {
                    "message": f"Unexpected error: {str(e)}",
                    "type": "service_error",
                }
            }

    async def call_model(self, translated_request: Dict[str, Any]) -> Dict[str, Any]:
        """High-level method to call the model with callback support"""
        return await super().call_model(translated_request)
    
    async def supports_tools(self) -> bool:
        """Check if this model supports tool/function calling"""
        # To be overridden by specific providers
        return False
