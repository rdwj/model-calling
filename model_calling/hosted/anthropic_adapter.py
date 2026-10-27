"""
Anthropic API adapter
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


class AnthropicAdapter(HostedProviderAdapter):
    """Adapter for Anthropic API"""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize Anthropic adapter"""
        model_config["provider_name"] = "anthropic"
        super().__init__(model_config)
        self.version = model_config.get("version", "2023-06-01")
    
    def _get_api_key(self) -> str:
        """Get Anthropic API key from environment"""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not found in environment")
        return api_key
    
    def _get_base_url(self) -> str:
        """Get Anthropic API base URL"""
        return os.environ.get("ANTHROPIC_API_BASE", "https://api.anthropic.com/v1/messages")
    
    async def _make_request_headers(self) -> Dict[str, str]:
        """Create headers for the Anthropic API request"""
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.version
        }
    
    async def translate_request(self, openai_format_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI format to Anthropic format"""
        messages = openai_format_request.get("messages", [])
        
        # Extract system message if present
        system = ""
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                user_messages.append(msg)
                
        # Convert to Anthropic format
        anthropic_request = {
            "model": self.model_name,
            "max_tokens": openai_format_request.get("max_tokens", 1024),
            "stream": openai_format_request.get("stream", False)
        }
        
        # Add system if present
        if system:
            anthropic_request["system"] = system
            
        # Add temperature if present
        if "temperature" in openai_format_request:
            anthropic_request["temperature"] = openai_format_request["temperature"]
        
        # Convert messages
        anthropic_messages = []
        
        # Handle tools/functions
        if "tools" in openai_format_request and await self.supports_tools():
            tools = []
            for tool in openai_format_request["tools"]:
                if tool["type"] == "function":
                    tools.append({
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "input_schema": tool["function"]["parameters"]
                    })
            
            if tools:
                anthropic_request["tools"] = tools
        
        # Convert messages
        processed_messages = []
        for msg in user_messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                processed_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                # Handle function calling in assistant messages
                if "function_call" in msg:
                    tool_call = {
                        "id": msg.get("function_call", {}).get("name", "tool_call"),
                        "name": msg["function_call"]["name"],
                        "input": json.loads(msg["function_call"]["arguments"])
                    }
                    
                    processed_messages.append({
                        "role": "assistant", 
                        "content": content or "",
                        "tool_calls": [tool_call]
                    })
                else:
                    processed_messages.append({"role": "assistant", "content": content})
            elif role == "function" or role == "tool":
                processed_messages.append({
                    "role": "tool",
                    "name": msg.get("name", "unknown_tool"),
                    "content": content
                })
        
        anthropic_request["messages"] = processed_messages
        
        return anthropic_request
    
    async def translate_response(self, anthropic_response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Anthropic response to OpenAI format"""
        if "error" in anthropic_response:
            return anthropic_response
            
        # Extract content and tool calls
        content = anthropic_response.get("content", [])
        
        # For Claude, content is an array of blocks
        text_content = ""
        tool_calls = []
        
        for block in content:
            if block["type"] == "text":
                text_content += block["text"]
            elif block["type"] == "tool_call":
                tool_calls.append({
                    "name": block["name"],
                    "arguments": json.dumps(block["input"])
                })
        
        # Check if there are tool calls
        if tool_calls:
            finish_reason = "function_call"
            # Use the first tool call for function_call in OpenAI format
            function_call = {
                "name": tool_calls[0]["name"],
                "arguments": tool_calls[0]["arguments"]
            }
        else:
            finish_reason = anthropic_response.get("stop_reason", "stop")
            function_call = None
            
        # Create OpenAI-compatible response
        return create_openai_response(
            model=f"anthropic/{self.model_name}",
            content=text_content,
            function_calls=[function_call] if function_call else None,
            finish_reason=finish_reason
        )
    
    async def translate_streaming_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Translate an Anthropic streaming chunk to OpenAI format"""
        # Extract content from the chunk
        delta = chunk.get("delta", {})
        text = delta.get("text", "")
        
        # Check if this is a tool call
        tool_calls = []
        if "tool_calls" in delta:
            for tool_call in delta["tool_calls"]:
                tool_calls.append({
                    "name": tool_call["name"],
                    "arguments": json.dumps(tool_call.get("input", {}))
                })
        
        # Determine if this is the final chunk
        is_final = chunk.get("type") == "message_stop"
        
        # Convert to OpenAI streaming format
        return create_openai_streaming_chunk(
            model=f"anthropic/{self.model_name}",
            content=text,
            is_final=is_final,
            function_calls=tool_calls if tool_calls else None
        )
    
    async def supports_tools(self) -> bool:
        """Check if this Anthropic model supports function calling"""
        tool_enabled_models = [
            "claude-3-opus", "claude-3-sonnet", "claude-3-haiku", 
            "claude-3-5-sonnet"
        ]
        
        return any(model in self.model_name for model in tool_enabled_models)


# Available Anthropic models and capabilities
ANTHROPIC_MODELS = {
    "claude-3-opus-20240229": {
        "supports_function_call": True,
        "supports_vision": True
    },
    "claude-3-sonnet-20240229": {
        "supports_function_call": True,
        "supports_vision": True
    },
    "claude-3-haiku-20240307": {
        "supports_function_call": True,
        "supports_vision": True
    },
    "claude-3-5-sonnet-20240620": {
        "supports_function_call": True,
        "supports_vision": True
    },
    "claude-2.1": {
        "supports_function_call": False,
        "supports_vision": False
    },
    "claude-2.0": {
        "supports_function_call": False,
        "supports_vision": False
    },
    "claude-instant-1.2": {
        "supports_function_call": False,
        "supports_vision": False
    }
}
