"""
Adapter for Ollama models.
"""
import json
import logging
import time
import sys
from typing import Dict, List, Any, Optional, Tuple
import httpx

from model_calling.adapter import ModelAdapter, StreamingAdapter
from model_calling.thinking import ThinkingStyle

# Configure logging - you can adjust this based on your deployment environment
logger = logging.getLogger(__name__)


class OllamaAdapter(ModelAdapter, StreamingAdapter):
    """Adapter for Ollama models"""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize the adapter with model configuration"""
        super().__init__(model_config)
        self.base_url = model_config.get("base_url", "http://localhost:11434")
        # Extract model name from the format "ollama/model-name"
        full_name = model_config.get("model_name", "")
        self.model_name = full_name.split("/")[-1] if "/" in full_name else full_name
        
        # Store all capabilities
        self.supports_function_call = model_config.get("supports_function_call", False)
        self.supports_streaming = model_config.get("supports_streaming", True)
        self.supports_system_messages = model_config.get("supports_system_messages", True)
        
        # Check for vision capabilities - explicitly enable for Llava models
        self.supports_vision = model_config.get("supports_vision", False)
        if "llava" in self.model_name.lower():
            self.supports_vision = True
            logger.info(f"Vision capabilities enabled for {self.model_name}")
            
        self.supports_json_mode = model_config.get("supports_json_mode", False)
        
        # Check for reasoning capabilities - explicitly enable for Phi4 models
        self.supports_reasoning = model_config.get("supports_reasoning", False)
        if "phi4" in self.model_name.lower() or "phi-4" in self.model_name.lower():
            self.supports_reasoning = True
            logger.info(f"Reasoning capabilities enabled for {self.model_name}")
            
        # Check for thinking capabilities - enable for supported models
        self.thinking_style = model_config.get("thinking_style", ThinkingStyle.NONE)
        if "granite3.3" in self.model_name.lower():
            self.thinking_style = ThinkingStyle.GRANITE
            logger.info(f"Granite-style thinking mode enabled for {self.model_name}")
    
    async def supports_tools(self) -> bool:
        """Check if this Ollama model supports tool/function calling"""
        return self.supports_function_call
    
    async def detect_tool_call_support(self) -> Tuple[bool, str]:
        """Test if this model actually supports tool calling with a simple test
        
        Returns:
            Tuple[bool, str]: (supports_tool_calls, detailed_log)
        """
        # Simple weather function for testing
        weather_tool = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }]
        
        # Create a simple request to test tool calling
        test_request = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": "What's the weather like in Tokyo?"}
            ],
            "tools": weather_tool,
            "stream": False,
            "temperature": 0.1  # Low temperature for more deterministic results
        }
        
        logs = []
        logs.append(f"Testing tool call support for {self.model_name}...")
        
        # We'll try multiple times with increasing timeouts to handle reliability issues
        timeouts = [60.0, 120.0, 180.0]  # Try with 1, 2, and 3 minute timeouts
        
        for attempt, timeout in enumerate(timeouts):
            try:
                logs.append(f"Attempt {attempt+1} with {timeout}s timeout")
                
                # Translate the request to Ollama format
                translated_request = await self.translate_request(test_request)
                
                # Call the API with the current timeout
                start_time = time.time()
                raw_response = await self._call_model_api(translated_request, timeout=timeout)
                elapsed_time = time.time() - start_time
                logs.append(f"Response received in {elapsed_time:.2f} seconds")
                
                # Check the response
                message = raw_response.get("message", {})
                
                # Log both the request and response for debugging
                logs.append(f"Tool call test request: {json.dumps(translated_request)[:300]}...")
                logs.append(f"Tool call test response: {json.dumps(raw_response)[:300]}...")
                
                # Check if the model returned a tool call
                if "tool_calls" in message:
                    tool_calls = message["tool_calls"]
                    logs.append(f"Model returned tool calls: {json.dumps(tool_calls)}")
                    
                    # Basic validation of the tool call
                    for tool_call in tool_calls:
                        if "function" in tool_call:
                            function_call = tool_call["function"]
                            if function_call.get("name") == "get_weather":
                                try:
                                    # Handle both string and dict arguments
                                    args = function_call.get("arguments", {})
                                    if isinstance(args, str):
                                        args = json.loads(args)
                                        
                                    if "location" in args:
                                        logs.append(f"Valid tool call detected with location: {args['location']}")
                                        return True, "\n".join(logs)
                                except Exception as e:
                                    logs.append(f"Error parsing function arguments: {e}")
                    
                    logs.append("Tool calls found but not valid for the weather function")
                    return False, "\n".join(logs)
                else:
                    # Check content for attempts at function calling
                    content = message.get("content", "")
                    logs.append(f"No tool calls in response. Content: {content[:200]}...")
                    
                    # Check if content mentions ignoring the functions or shows attempts
                    if "function" in content.lower() or "tool" in content.lower():
                        logs.append("Model may be attempting to use functions in text but doesn't support proper tool calls")
                    
                    # If not the last attempt, continue to the next attempt
                    if attempt < len(timeouts) - 1:
                        logs.append("Trying again with longer timeout...")
                        continue
                    
                    return False, "\n".join(logs)
            
            except Exception as e:
                logs.append(f"Error in attempt {attempt+1}: {e}")
                # If not the last attempt, continue to the next attempt
                if attempt < len(timeouts) - 1:
                    logs.append("Trying again with longer timeout...")
                    continue
                
                return False, "\n".join(logs)
        
        # If we've exhausted all attempts
        logs.append("All attempts failed to detect tool call support")
        return False, "\n".join(logs)
    
    async def translate_request(self, openai_format_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI format request to Ollama format"""
        # Extract the messages
        messages = openai_format_request.get("messages", [])
        
        # Add thinking mode control message if supported
        if self.thinking_style != ThinkingStyle.NONE:
            if self.thinking_style == ThinkingStyle.GRANITE:
                messages.insert(0, {
                    "role": "system",
                    "content": "You are a helpful AI assistant that thinks through problems step by step. First think through the problem in <think></think> tags, then provide your response in <response></response> tags."
                })
            # Add other thinking styles as needed
        
        # Handle JSON mode if requested
        response_format = openai_format_request.get("response_format", {})
        if response_format and response_format.get("type") == "json_object" and self.supports_json_mode:
            # Add a system message or update existing one to request JSON
            json_instruction = "You must respond with a valid JSON object only, with no additional text or explanation."
            
            # Check if there's already a system message
            has_system = False
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    has_system = True
                    messages[i]["content"] = messages[i].get("content", "") + "\n" + json_instruction
                    break
            
            # If no system message, add one
            if not has_system:
                messages.insert(0, {
                    "role": "system",
                    "content": json_instruction
                })
        
        # Basic request structure
        ollama_request = {
            "model": self.model_name,
            "messages": messages,
            "stream": openai_format_request.get("stream", False)
        }
        
        # Handle tools if model supports it
        if "tools" in openai_format_request and self.supports_function_call:
            ollama_tools = []
            for tool in openai_format_request["tools"]:
                if tool["type"] == "function":
                    ollama_tools.append({
                        "type": "function",
                        "function": tool["function"]
                    })
            
            if ollama_tools:
                ollama_request["tools"] = ollama_tools
        
        # Map common parameters
        if "temperature" in openai_format_request:
            ollama_request["temperature"] = openai_format_request["temperature"]
        
        if "top_p" in openai_format_request:
            ollama_request["top_p"] = openai_format_request["top_p"]
        
        return ollama_request
    
    def _extract_thinking_content(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract thinking and response content from the model output"""
        import re
        
        # Extract thinking content if present
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else None
        
        # Extract response content if present
        response_match = re.search(r'<response>(.*?)</response>', content, re.DOTALL)
        response = response_match.group(1).strip() if response_match else content
        
        return thinking, response
    
    async def translate_response(self, ollama_response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Ollama response to OpenAI format"""
        if "error" in ollama_response:
            return ollama_response
        
        # Extract content and token counts
        message = ollama_response.get("message", {})
        content = message.get("content", "")
        
        # Extract thinking and response content
        thinking, response = self._extract_thinking_content(content)
        
        # Handle function calls if present
        function_calls = None
        if "tool_calls" in message:
            function_calls = []
            for tool_call in message["tool_calls"]:
                if "function" in tool_call:
                    function_calls.append({
                        "id": f"call_{len(function_calls)}",
                        "type": "function",
                        "function": {
                            "name": tool_call["function"]["name"],
                            "arguments": json.dumps(tool_call["function"]["arguments"])
                        }
                    })
        
        # Get token counts
        prompt_tokens = ollama_response.get("prompt_eval_count", 0)
        completion_tokens = ollama_response.get("eval_count", 0)
        
        logger.debug(f"Token counts from Ollama - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {prompt_tokens + completion_tokens}")
        
        # Build response in OpenAI format
        response_obj = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": f"ollama/{self.model_name}",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": ollama_response.get("done_reason", "stop")
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
        
        # Add thinking content if present
        if thinking:
            response_obj["choices"][0]["message"]["thinking"] = thinking
        
        # Add function calls if present
        if function_calls:
            response_obj["choices"][0]["message"]["tool_calls"] = function_calls
        
        return response_obj
    
    async def translate_streaming_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Ollama streaming chunk to OpenAI format"""
        message = chunk.get("message", {})
        is_final = chunk.get("done", False)
        content = message.get("content", "")
        
        # Extract thinking and response content
        thinking, response = self._extract_thinking_content(content)
        
        # For function calling in streaming mode
        function_calls = None
        if "tool_calls" in message:
            function_calls = []
            for tool_call in message["tool_calls"]:
                if "function" in tool_call:
                    function_calls.append({
                        "id": f"call_{len(function_calls)}",
                        "type": "function",
                        "function": {
                            "name": tool_call["function"]["name"],
                            "arguments": json.dumps(tool_call["function"]["arguments"])
                        }
                    })
        
        chunk_data = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": f"ollama/{self.model_name}",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant" if is_final else None,
                    "content": response
                },
                "finish_reason": chunk.get("done_reason", "stop") if is_final else None
            }]
        }
        
        # Add thinking content if present
        if thinking:
            chunk_data["choices"][0]["delta"]["thinking"] = thinking
        
        # Add function calls if present and final
        if function_calls and is_final:
            chunk_data["choices"][0]["delta"]["tool_calls"] = function_calls
        
        return chunk_data
    
    def _log_request_response(self, request: Dict[str, Any], response: Dict[str, Any], is_tool_call: bool = False):
        """Log request and response details for debugging"""
        request_to_log = request.copy()
        # Truncate long message contents for readability
        if "messages" in request_to_log:
            for i, msg in enumerate(request_to_log["messages"]):
                if "content" in msg and isinstance(msg["content"], str) and len(msg["content"]) > 200:
                    request_to_log["messages"][i]["content"] = msg["content"][:200] + "..."
        
        log_prefix = "[TOOL CALL] " if is_tool_call else ""
        
        logger.debug(f"{log_prefix}Request to Ollama ({self.model_name}): {json.dumps(request_to_log)}")
        
        if is_tool_call:
            # For tool calls, log more details
            logger.info(f"Tool call request to {self.model_name}")
            # Log if tools are in the request
            if "tools" in request:
                tool_names = [t.get("function", {}).get("name", "unknown") for t in request.get("tools", [])]
                logger.info(f"Requesting tools: {', '.join(tool_names)}")
            
            # Check response format
            message = response.get("message", {})
            if "tool_calls" in message:
                tool_calls = message["tool_calls"]
                logger.info(f"Model {self.model_name} returned tool calls: {json.dumps(tool_calls)}")
            elif "content" in message:
                # Log the first part of the content to see if it contains function calling language
                content = message.get("content", "")
                logger.info(f"Model {self.model_name} did not return formal tool calls")
                if content:
                    # Look for patterns suggesting the model is trying to call a function but not in the right format
                    function_pattern = r"function\s*\w+\s*\("
                    if re.search(function_pattern, content):
                        logger.warning(f"Model {self.model_name} may be attempting to call functions in text: {content[:200]}...")
            
            if "error" in response:
                logger.error(f"Ollama returned error: {response['error']}")
        else:
            logger.debug(f"Response from Ollama: {json.dumps(response)[:500]}...")
    
    async def _call_model_api(self, translated_request: Dict[str, Any], timeout: float = 300.0) -> Dict[str, Any]:
        """Call the Ollama API and return the raw response"""
        is_streaming = translated_request.get("stream", False)
        endpoint = f"{self.base_url}/api/chat"
        
        # Detect if this is a tool call request
        is_tool_call = "tools" in translated_request and self.supports_function_call
        log_level = logging.INFO if is_tool_call else logging.DEBUG
        
        logger.log(log_level, f"Sending request to Ollama endpoint: {endpoint}")
        
        async with httpx.AsyncClient() as client:
            if is_streaming:
                # If streaming, we'll collect all chunks and combine
                all_chunks = []
                messages = translated_request.get("messages", [])
                
                async with client.stream("POST", endpoint, json=translated_request, timeout=timeout) as response:
                    async for chunk in response.aiter_lines():
                        if not chunk.strip():
                            continue
                        try:
                            chunk_data = json.loads(chunk)
                            # Handle callbacks and translation
                            translated_chunk = await self.handle_streaming_chunk(
                                chunk_data, messages, **translated_request
                            )
                            all_chunks.append(chunk_data)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse Ollama chunk: {chunk}")
                
                # The final chunk should contain token counts
                final_chunk = all_chunks[-1] if all_chunks else {}
                
                # Combine chunks for a complete response
                final_content = ""
                tool_calls = None
                
                for chunk in all_chunks:
                    message = chunk.get("message", {})
                    if "content" in message:
                        final_content += message.get("content", "")
                    
                    # Take the first tool_calls we find
                    if "tool_calls" in message and not tool_calls:
                        tool_calls = message["tool_calls"]
                
                # Create a final combined response with token counts
                final_response = {
                    "model": self.model_name,
                    "message": {
                        "role": "assistant",
                        "content": final_content
                    },
                    "done": True,
                    "done_reason": "stop",
                    # Include token counts from the final chunk
                    "prompt_eval_count": final_chunk.get("prompt_eval_count", 0),
                    "eval_count": final_chunk.get("eval_count", 0)
                }
                
                if tool_calls:
                    final_response["message"]["tool_calls"] = tool_calls
                
                if is_tool_call:
                    self._log_request_response(translated_request, final_response, is_tool_call=True)
                
                return final_response
            else:
                # For non-streaming requests, make a single call
                response = await client.post(endpoint, json=translated_request, timeout=timeout)
                response_data = response.json()
                
                # Log request and response for debugging
                if is_tool_call:
                    self._log_request_response(translated_request, response_data, is_tool_call=True)
                
                return response_data
    
    async def call_model(self, translated_request: Dict[str, Any]) -> Dict[str, Any]:
        """Call the Ollama API and return the response"""
        return await self._call_model_api(translated_request)


# We no longer determine function call capability based on model name
# Instead, we test it directly using detect_tool_call_support
