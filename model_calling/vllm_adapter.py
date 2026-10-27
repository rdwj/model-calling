"""
Adapter for vLLM-deployed models.
"""
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import httpx

from model_calling.adapter import ModelAdapter, StreamingAdapter, create_openai_response, create_openai_streaming_chunk

logger = logging.getLogger(__name__)


class VLLMAdapter(ModelAdapter, StreamingAdapter):
    """Adapter for vLLM-deployed models"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.base_url = model_config.get("base_url", "http://localhost:8000")
        # Extract model name from the format "vllm/model-name"
        full_name = model_config.get("model_name", "")
        self.model_name = full_name.split("/")[-1] if "/" in full_name else full_name
        # vLLM can be configured to use OpenAI-compatible API format
        self.openai_compatible = model_config.get("openai_compatible", True)
        self.supports_function_call = model_config.get("supports_function_call", False)
    
    async def supports_tools(self) -> bool:
        """Check if this vLLM model supports tool/function calling"""
        return self.supports_function_call and self.openai_compatible
    
    async def translate_request(self, openai_format_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert request format if needed (may be already OpenAI compatible)"""
        if self.openai_compatible:
            # vLLM with OpenAI compatible API can use the request directly
            # Just update the model name
            modified_request = openai_format_request.copy()
            modified_request["model"] = self.model_name
            
            # Handle the case where tools are not supported
            if not self.supports_function_call and "tools" in modified_request:
                del modified_request["tools"]
                
            return modified_request
        else:
            # For custom vLLM endpoints, we might need to translate
            # This is a simplified version, real implementation would need more detail
            vllm_request = {
                "model": self.model_name,
                "prompt": self._convert_messages_to_prompt(openai_format_request.get("messages", [])),
                "stream": openai_format_request.get("stream", False)
            }
            
            # Map common parameters
            if "temperature" in openai_format_request:
                vllm_request["temperature"] = openai_format_request["temperature"]
            
            if "top_p" in openai_format_request:
                vllm_request["top_p"] = openai_format_request["top_p"]
                
            if "max_tokens" in openai_format_request:
                vllm_request["max_tokens"] = openai_format_request["max_tokens"]
                
            return vllm_request
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert chat messages to a text prompt for non-OpenAI compatible endpoints"""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"<|system|>\n{content}")
            elif role == "user":
                prompt_parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}")
            elif role == "tool":
                prompt_parts.append(f"<|tool|>\n{content}")
                
        # Add the final assistant prefix to indicate where model should continue
        prompt_parts.append("<|assistant|>")
        
        return "\n".join(prompt_parts)
    
    async def translate_response(self, vllm_response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert vLLM response to OpenAI format if needed"""
        if self.openai_compatible:
            # For OpenAI-compatible endpoints, the response is already in the right format
            # We might just need to update the model name
            if "model" in vllm_response:
                vllm_response["model"] = f"vllm/{self.model_name}"
            return vllm_response
        else:
            # For custom vLLM endpoints, we convert the response
            return create_openai_response(
                model=f"vllm/{self.model_name}",
                content=vllm_response.get("text", ""),
                finish_reason=vllm_response.get("finish_reason", "stop")
            )
    
    async def translate_streaming_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Convert vLLM streaming chunk to OpenAI format if needed"""
        if self.openai_compatible:
            # For OpenAI-compatible endpoints, just update the model name
            if "model" in chunk:
                chunk["model"] = f"vllm/{self.model_name}"
            return chunk
        else:
            # For custom vLLM endpoints
            return create_openai_streaming_chunk(
                model=f"vllm/{self.model_name}",
                content=chunk.get("text", ""),
                is_final=chunk.get("done", False)
            )
    
    async def _call_model_api(self, translated_request: Dict[str, Any]) -> Dict[str, Any]:
        """Call the vLLM API and return the raw response"""
        is_streaming = translated_request.get("stream", False)
        
        if self.openai_compatible:
            endpoint = f"{self.base_url}/v1/chat/completions"
        else:
            endpoint = f"{self.base_url}/generate"
        
        async with httpx.AsyncClient() as client:
            if is_streaming:
                # Handle streaming responses
                all_chunks = []
                messages = translated_request.get("messages", [])
                
                async with client.stream("POST", endpoint, json=translated_request, timeout=60.0) as response:
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
                            logger.error(f"Failed to parse vLLM chunk: {chunk}")
                
                # For OpenAI-compatible endpoints, just return the last chunk
                if self.openai_compatible and all_chunks:
                    return all_chunks[-1]
                
                # For custom endpoints, combine chunks
                final_text = ""
                for chunk in all_chunks:
                    if "text" in chunk:
                        final_text += chunk["text"]
                
                return {
                    "text": final_text,
                    "done": True,
                    "finish_reason": "stop"
                }
            else:
                # For non-streaming responses
                response = await client.post(endpoint, json=translated_request, timeout=60.0)
                return response.json()

    async def call_model(self, translated_request: Dict[str, Any]) -> Dict[str, Any]:
        """High-level method to call the model with callback support"""
        return await super().call_model(translated_request)
