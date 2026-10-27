"""
Comprehensive capability testing module for LLM models.
Tests actual capabilities rather than relying on model name patterns.
"""
import json
import logging
import time
import re
from typing import Dict, List, Any, Optional, Tuple, Union

import httpx

logger = logging.getLogger(__name__)

class CapabilityTester:
    """Base class for testing various model capabilities"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize the capability tester
        
        Args:
            base_url: The base URL for the model API
        """
        self.base_url = base_url
        
    async def test_function_calling(self, model_name: str) -> Tuple[bool, str]:
        """Test if a model supports function calling
        
        Args:
            model_name: The name of the model to test
            
        Returns:
            Tuple[bool, str]: (supports_function_calling, detailed_log)
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
            "model": model_name,
            "messages": [
                {"role": "user", "content": "What's the weather like in Tokyo?"}
            ],
            "tools": weather_tool,
            "stream": False,
            "temperature": 0.1  # Low temperature for more deterministic results
        }
        
        logs = []
        logs.append(f"Testing function calling for {model_name}...")
        
        # Try with increasing timeouts
        timeouts = [60.0, 120.0]
        
        for attempt, timeout in enumerate(timeouts):
            try:
                logs.append(f"Attempt {attempt+1} with {timeout}s timeout")
                
                # Call the API with the current timeout
                start_time = time.time()
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/api/chat", 
                        json=test_request,
                        timeout=timeout
                    )
                    
                if response.status_code != 200:
                    logs.append(f"Error: {response.status_code} - {response.text}")
                    if attempt < len(timeouts) - 1:
                        logs.append("Trying again with longer timeout...")
                        continue
                    return False, "\n".join(logs)
                
                raw_response = response.json()
                elapsed_time = time.time() - start_time
                logs.append(f"Response received in {elapsed_time:.2f} seconds")
                
                # Check the response
                message = raw_response.get("message", {})
                
                # Log response for debugging
                logs.append(f"Response: {json.dumps(raw_response)[:300]}...")
                
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
        logs.append("All attempts failed to detect function calling support")
        return False, "\n".join(logs)
    
    async def test_json_mode(self, model_name: str) -> Tuple[bool, str]:
        """Test if a model supports JSON mode
        
        Args:
            model_name: The name of the model to test
            
        Returns:
            Tuple[bool, str]: (supports_json_mode, detailed_log)
        """
        test_request = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You must respond with valid JSON only."},
                {"role": "user", "content": "List the top 3 planets by size in our solar system, with their names and diameters."}
            ],
            "response_format": {"type": "json_object"},
            "stream": False,
            "temperature": 0.1
        }
        
        logs = []
        logs.append(f"Testing JSON mode for {model_name}...")
        
        try:
            # Call the API
            start_time = time.time()
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/chat", 
                    json=test_request,
                    timeout=60.0
                )
                
            if response.status_code != 200:
                logs.append(f"Error: {response.status_code} - {response.text}")
                return False, "\n".join(logs)
            
            raw_response = response.json()
            elapsed_time = time.time() - start_time
            logs.append(f"Response received in {elapsed_time:.2f} seconds")
            
            # Check the response
            message = raw_response.get("message", {})
            content = message.get("content", "")
            
            # Log response for debugging
            logs.append(f"Response content: {content[:300]}...")
            
            # Check if the content is valid JSON
            try:
                if not content.strip():
                    logs.append("Empty response content")
                    return False, "\n".join(logs)
                
                # Try to parse as JSON
                parsed_json = json.loads(content)
                
                # Check if it has expected structure (planets with sizes)
                if isinstance(parsed_json, dict) and any(key in content.lower() for key in ["planets", "jupiter", "saturn"]):
                    logs.append(f"Valid JSON response detected: {json.dumps(parsed_json)[:100]}...")
                    return True, "\n".join(logs)
                else:
                    logs.append(f"Response is valid JSON but doesn't match expected structure: {content[:100]}...")
                    return False, "\n".join(logs)
            except json.JSONDecodeError:
                logs.append(f"Response is not valid JSON: {content[:100]}...")
                return False, "\n".join(logs)
        
        except Exception as e:
            logs.append(f"Error testing JSON mode: {e}")
            return False, "\n".join(logs)
    
    async def test_reasoning(self, model_name: str) -> Tuple[bool, str]:
        """Test if a model supports reasoning capabilities
        
        Args:
            model_name: The name of the model to test
            
        Returns:
            Tuple[bool, str]: (supports_reasoning, detailed_log)
        """
        test_request = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": "If John is twice as old as Mary was when John was as old as Mary is now, and Mary is 24, how old is John?"}
            ],
            "stream": False,
            "temperature": 0.1
        }
        
        logs = []
        logs.append(f"Testing reasoning capabilities for {model_name}...")
        
        try:
            # Call the API
            start_time = time.time()
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/chat", 
                    json=test_request,
                    timeout=60.0
                )
                
            if response.status_code != 200:
                logs.append(f"Error: {response.status_code} - {response.text}")
                return False, "\n".join(logs)
            
            raw_response = response.json()
            elapsed_time = time.time() - start_time
            logs.append(f"Response received in {elapsed_time:.2f} seconds")
            
            # Check the response
            message = raw_response.get("message", {})
            content = message.get("content", "")
            
            # Log response for debugging
            logs.append(f"Response content: {content[:300]}...")
            
            # Check for signs of step-by-step reasoning
            step_by_step_indicators = [
                "first", "step", "let's", "calculate", "equation", "let", "set up", 
                "solve", "therefore", "thus", "since", "we know", "so", "then"
            ]
            
            # Check for the correct answer (36) with explanation
            has_correct_answer = "36" in content and any(indicator in content.lower() for indicator in step_by_step_indicators)
            has_reasoning = len(content) > 100 and any(indicator in content.lower() for indicator in step_by_step_indicators)
            
            if has_correct_answer:
                logs.append("Model provided correct answer (36) with reasoning")
                return True, "\n".join(logs)
            elif has_reasoning:
                logs.append("Model demonstrated reasoning capabilities but may not have the correct answer")
                return True, "\n".join(logs)
            else:
                logs.append("Model did not demonstrate sufficient reasoning capabilities")
                return False, "\n".join(logs)
        
        except Exception as e:
            logs.append(f"Error testing reasoning capabilities: {e}")
            return False, "\n".join(logs)
    
    async def test_thinking(self, model_name: str) -> Tuple[bool, str]:
        """Test if a model supports thinking capabilities (showing work)
        
        Args:
            model_name: The name of the model to test
            
        Returns:
            Tuple[bool, str]: (supports_thinking, detailed_log)
        """
        # Test for different thinking styles
        
        # First try Granite-style thinking with <think></think> tags
        granite_request = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant that thinks through problems step by step. First think through the problem in <think></think> tags, then provide your response in <response></response> tags."},
                {"role": "user", "content": "What is the square root of 841 and how did you figure it out?"}
            ],
            "stream": False,
            "temperature": 0.1
        }
        
        logs = []
        logs.append(f"Testing thinking capabilities for {model_name}...")
        
        try:
            # Call the API with Granite-style thinking
            logs.append("Testing Granite-style thinking (<think></think> tags)...")
            start_time = time.time()
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/chat", 
                    json=granite_request,
                    timeout=60.0
                )
                
            if response.status_code != 200:
                logs.append(f"Error: {response.status_code} - {response.text}")
            else:
                raw_response = response.json()
                elapsed_time = time.time() - start_time
                logs.append(f"Response received in {elapsed_time:.2f} seconds")
                
                # Check the response
                message = raw_response.get("message", {})
                content = message.get("content", "")
                
                # Log response for debugging
                logs.append(f"Response content: {content[:300]}...")
                
                # Check for thinking tags
                think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                response_match = re.search(r'<response>(.*?)</response>', content, re.DOTALL)
                
                if think_match and response_match:
                    thinking = think_match.group(1).strip()
                    response_content = response_match.group(1).strip()
                    
                    logs.append(f"Thinking section found: {thinking[:100]}...")
                    logs.append(f"Response section found: {response_content[:100]}...")
                    return True, "\n".join(logs)
            
            # If Granite-style thinking didn't work, try general step-by-step thinking
            logs.append("Testing general step-by-step thinking...")
            general_request = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "When solving problems, always show your work step by step."},
                    {"role": "user", "content": "What is the square root of 841 and how did you figure it out?"}
                ],
                "stream": False,
                "temperature": 0.1
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/chat", 
                    json=general_request,
                    timeout=60.0
                )
                
            if response.status_code != 200:
                logs.append(f"Error: {response.status_code} - {response.text}")
                return False, "\n".join(logs)
            
            raw_response = response.json()
            
            # Check the response
            message = raw_response.get("message", {})
            content = message.get("content", "")
            
            # Check for evidence of step-by-step thinking
            thinking_indicators = [
                "step", "first", "next", "then", "finally", 
                "approach", "method", "calculate", "compute",
                "square", "√", "sqrt", "root"
            ]
            
            has_correct_answer = "29" in content
            has_thinking = any(indicator in content.lower() for indicator in thinking_indicators)
            has_steps = len(content) > 100 and content.count("\n") >= 3
            
            if has_thinking and has_steps and has_correct_answer:
                logs.append("Model demonstrated general thinking capabilities")
                return True, "\n".join(logs)
            else:
                logs.append("Model did not demonstrate sufficient thinking capabilities")
                return False, "\n".join(logs)
        
        except Exception as e:
            logs.append(f"Error testing thinking capabilities: {e}")
            return False, "\n".join(logs)
    
    async def test_vision(self, model_name: str) -> Tuple[bool, str]:
        """Test if a model supports vision capabilities
        
        Note: This is just a placeholder since we can't easily test vision
        without providing an image. For actual implementation, you'd need
        to include a test image.
        
        Args:
            model_name: The name of the model to test
            
        Returns:
            Tuple[bool, str]: (supports_vision, detailed_log)
        """
        logs = []
        logs.append(f"Vision capabilities for {model_name} can only be determined by model family")
        
        # Check model name for vision indicators
        vision_models = ["llava", "bakllava", "vision", "qwen-vl", "cogvlm", "clip"]
        has_vision_name = any(vision_model in model_name.lower() for vision_model in vision_models)
        
        if has_vision_name:
            logs.append(f"Model name {model_name} suggests vision capabilities")
            return True, "\n".join(logs)
        else:
            logs.append(f"Model name {model_name} does not suggest vision capabilities")
            return False, "\n".join(logs)
    
    async def check_model_exists(self, model_name: str) -> bool:
        """Check if a model exists in Ollama
        
        Args:
            model_name: The name of the model to check
            
        Returns:
            bool: True if the model exists, False otherwise
        """
        try:
            # Ollama API uses POST for show endpoint, not GET
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/show",
                    json={"name": model_name},
                    timeout=10.0
                )
                
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Error checking if model {model_name} exists: {e}")
            return False
    
    async def test_all_capabilities(self, model_name: str) -> Dict[str, Any]:
        """Test all capabilities for a model
        
        Args:
            model_name: The name of the model to test
            
        Returns:
            Dict[str, Any]: Dictionary of capability test results
        """
        logger.info(f"Testing all capabilities for {model_name}...")
        start_time = time.time()
        
        # First check if the model exists
        model_exists = await self.check_model_exists(model_name)
        if not model_exists:
            logger.warning(f"Model {model_name} does not exist in Ollama")
            return {
                "supports_function_call": False,
                "supports_streaming": True,
                "supports_system_messages": True,
                "supports_vision": False,
                "supports_json_mode": False,
                "supports_reasoning": False,
                "supports_thinking": False,
                "verified": False,
                "last_tested": int(time.time()),
                "error": "Model not found"
            }
        
        # Initialize results with default capabilities
        results = {
            "supports_streaming": True,  # Most models support streaming
            "supports_system_messages": True,  # Most models support system messages
            "supports_vision": False,  # Default to no vision support
            "supports_json_mode": False,  # Default to no JSON mode support
            "supports_reasoning": False,  # Default to no reasoning support
            "supports_thinking": False,  # Default to no thinking support
            "verified": True,  # We're verifying the model
            "last_tested": int(time.time())  # Current timestamp
        }
        
        # Test all capabilities sequentially
        # Test function calling first as it's most important
        try:
            supports_function_call, fn_call_log = await self.test_function_calling(model_name)
            results["supports_function_call"] = supports_function_call
            logger.info(f"Function calling test for {model_name}: {supports_function_call}")
        except Exception as e:
            logger.warning(f"Function calling test failed for {model_name}: {e}")
            results["supports_function_call"] = False
        
        # Test JSON mode if applicable (more capable models)
        try:
            if results["supports_function_call"] or any(model in model_name.lower() for model in ["llama3", "mistral", "mixtral", "claude", "gpt"]):
                supports_json, json_log = await self.test_json_mode(model_name)
                results["supports_json_mode"] = supports_json
                logger.info(f"JSON mode test for {model_name}: {supports_json}")
        except Exception as e:
            logger.warning(f"JSON mode test failed for {model_name}: {e}")
            results["supports_json_mode"] = False
        
        # Test reasoning if applicable
        try:
            if results["supports_function_call"] or any(model in model_name.lower() for model in ["llama3", "mistral", "mixtral", "claude", "gpt", "phi"]):
                supports_reasoning, reasoning_log = await self.test_reasoning(model_name)
                results["supports_reasoning"] = supports_reasoning
                logger.info(f"Reasoning test for {model_name}: {supports_reasoning}")
        except Exception as e:
            logger.warning(f"Reasoning test failed for {model_name}: {e}")
            results["supports_reasoning"] = False
        
        # Test vision if applicable
        try:
            if any(vision_model in model_name.lower() for vision_model in ["llava", "bakllava", "vision"]):
                supports_vision, vision_log = await self.test_vision(model_name)
                results["supports_vision"] = supports_vision
                logger.info(f"Vision test for {model_name}: {supports_vision}")
        except Exception as e:
            logger.warning(f"Vision test failed for {model_name}: {e}")
            # Vision is still false by default
        
        # Test thinking capabilities
        try:
            if results["supports_reasoning"] or "granite" in model_name.lower():
                supports_thinking, thinking_log = await self.test_thinking(model_name)
                results["supports_thinking"] = supports_thinking
                logger.info(f"Thinking test for {model_name}: {supports_thinking}")
        except Exception as e:
            logger.warning(f"Thinking test failed for {model_name}: {e}")
            # Thinking is still false by default
        
        # Log completion
        elapsed_time = time.time() - start_time
        logger.info(f"Completed capability testing for {model_name} in {elapsed_time:.2f} seconds")
        logger.info(f"Results: {results}")
        
        results["last_tested"] = int(time.time())
        
        # We don't need to create detailed logs as we've already logged everything
        # and the variable names might not exist if tests weren't run
        
        return results

# Create singleton instance
capability_tester = CapabilityTester()
