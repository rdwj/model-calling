"""
Test configuration and utilities.
"""

import os
import json
import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define fallback models in case auto-configuration hasn't run
FALLBACK_FUNCTION_CALLING = [
    "mistral-small3.1:24b",  # Larger model with good function calling
    "granite3-dense:8b",  # Medium-sized model
    "llama3:latest",  # Alternative model family
]

FALLBACK_BASIC_MODELS = [
    "mistral:latest",  # Basic model for simple tests
    "llama3:latest",  # Alternative model family
]

# Try to import models from auto-generated file
try:
    from .models_available import FUNCTION_CALLING_MODELS, BASIC_MODELS
except ImportError:
    # Use fallback models if auto-config hasn't been run yet
    FUNCTION_CALLING_MODELS = FALLBACK_FUNCTION_CALLING
    BASIC_MODELS = FALLBACK_BASIC_MODELS

# Configuration for tests
TEST_CONFIG = {
    "service_url": os.environ.get("MODEL_CALLING_URL", "http://localhost:8000"),
    "ollama_models": {
        # Models that support function calling
        "function_calling": FUNCTION_CALLING_MODELS[
            :1
        ],  # Only use the first model to speed up tests
        # Models for basic tests
        "basic": BASIC_MODELS[:1],  # Only use the first model to speed up tests
    },
}

# Test data
TEST_MESSAGES = [{"role": "user", "content": "What is the capital of France?"}]

TEST_SYSTEM_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant that provides concise answers."},
    {"role": "user", "content": "What is the capital of France?"},
]

WEATHER_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
        },
    }
]

CALCULATOR_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to calculate",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]

MULTIPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to calculate",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


# Mock function implementations
def get_weather(location):
    """Mock weather function"""
    return {
        "location": location,
        "temperature": 22,
        "unit": "celsius",
        "description": "Partly cloudy",
    }


def calculate(expression):
    """Mock calculator function"""
    try:
        # Basic safety check - only allow simple arithmetic
        allowed_chars = set("0123456789+-*/() .")
        if not all(c in allowed_chars for c in expression):
            return {"error": "Invalid expression, only basic arithmetic allowed"}

        # Safely evaluate expression
        result = eval(expression, {"__builtins__": {}})
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


# Export these for other test files to use
__all__ = [
    "TEST_CONFIG",
    "TEST_MESSAGES",
    "TEST_SYSTEM_MESSAGES",
    "WEATHER_TOOL",
    "CALCULATOR_TOOL",
    "MULTIPLE_TOOLS",
    "get_weather",
    "calculate",
]
