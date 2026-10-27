"""
Base adapter classes for translating between OpenAI API format and model-specific formats.
"""
import time
import json
from abc import ABC, abstractmethod
from functools import wraps
from typing import Dict, List, Any, Optional, Sequence, Callable, TypeVar, Awaitable

from model_calling.callbacks import Callback


T = TypeVar('T')


def with_callbacks(f: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Decorator to add callback support to model calls"""
    @wraps(f)
    async def wrapped(self, translated_request: Dict[str, Any], *args, **kwargs) -> T:
        messages = translated_request.get("messages", [])
        
        # Call all callbacks before the request
        for callback in self.callbacks:
            await callback.on_request_start(self.model_name, messages, **translated_request)
        
        try:
            # Make the actual API call
            response = await f(self, translated_request, *args, **kwargs)
            
            # Call all callbacks after the request
            for callback in self.callbacks:
                await callback.on_request_end(self.model_name, messages, response, **translated_request)
            
            return response
        except Exception as e:
            # Call error callbacks
            for callback in self.callbacks:
                await callback.on_error(self.model_name, messages, e, **translated_request)
            raise
    
    return wrapped


class ModelAdapter(ABC):
    """Base class for all model adapters"""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize the adapter with model-specific configuration."""
        self.model_config = model_config
        self.base_url = model_config.get("base_url")
        self.model_name = model_config.get("model_name")
        self.callbacks: List[Callback] = model_config.get("callbacks", [])
    
    @abstractmethod
    async def translate_request(self, openai_format_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI format request to model-specific format"""
        pass
    
    @abstractmethod
    async def translate_response(self, model_response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert model-specific response to OpenAI format"""
        pass

    @abstractmethod
    async def _call_model_api(self, translated_request: Dict[str, Any]) -> Dict[str, Any]:
        """Send the request to the model backend and return the raw response.
        
        This is the core API call that each adapter must implement. It should:
        1. Make the API call to their backend
        2. Handle any model-specific error cases
        3. Return the raw response
        
        The base class handles:
        - Callback notifications
        - Generic error handling
        - Response translation
        """
        pass

    async def call_model(self, translated_request: Dict[str, Any]) -> Dict[str, Any]:
        """High-level method to call the model with callback support."""
        messages = translated_request.get("messages", [])
        
        # Call all callbacks before the request
        for callback in self.callbacks:
            await callback.on_request_start(self.model_name, messages, **translated_request)
        
        try:
            # Make the actual API call
            response = await self._call_model_api(translated_request)
            
            # Call all callbacks after the request
            for callback in self.callbacks:
                await callback.on_request_end(self.model_name, messages, response, **translated_request)
            
            return response
        except Exception as e:
            # Call error callbacks
            for callback in self.callbacks:
                await callback.on_error(self.model_name, messages, e, **translated_request)
            raise

    async def supports_tools(self) -> bool:
        """Check if the model supports tool/function calling"""
        return False


class StreamingAdapter(ABC):
    """Mixin for adapters that support streaming responses"""
    
    @abstractmethod
    async def translate_streaming_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a streaming chunk from model-specific to OpenAI format"""
        pass

    async def handle_streaming_chunk(self, chunk: Dict[str, Any], messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Handle a streaming chunk, including callback notifications
        
        Args:
            chunk: Raw chunk from the model
            messages: Original request messages
            **kwargs: Additional request parameters
        
        Returns:
            Translated chunk in OpenAI format
        """
        translated = await self.translate_streaming_chunk(chunk)
        
        # Call streaming callbacks if available
        if hasattr(self, 'callbacks'):
            for callback in self.callbacks:
                await callback.on_stream_chunk(self.model_name, messages, translated, **kwargs)
        
        return translated


def create_openai_response(
    model: str,
    content: str = "",
    function_calls: Optional[List[Dict[str, Any]]] = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0
) -> Dict[str, Any]:
    """Helper to create a standardized OpenAI-format response"""
    message = {"role": "assistant", "content": content}
    
    # Add function call if present
    if function_calls:
        message["function_call"] = function_calls[0]
        finish_reason = "function_call"
        
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens or (prompt_tokens + completion_tokens)
        }
    }


def create_openai_streaming_chunk(
    model: str,
    content: str,
    index: int = 0,
    is_final: bool = False,
    function_calls: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Helper to create a standardized OpenAI-format streaming chunk"""
    
    delta = {"role": "assistant"}
    finish_reason = None
    
    if content:
        delta["content"] = content
    
    if function_calls and is_final:
        delta["function_call"] = function_calls[0]
        finish_reason = "function_call"
    
    if is_final:
        finish_reason = finish_reason or "stop"
    
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": index,
            "delta": delta,
            "finish_reason": finish_reason
        }]
    }
