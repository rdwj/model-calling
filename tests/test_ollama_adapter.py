"""
Offline unit tests for the Ollama adapter.
"""

import logging

from model_calling.ollama_adapter import OllamaAdapter


def test_tool_call_logging_flags_function_written_as_text(caplog):
    """A model that writes a call as prose gets a warning, not a NameError."""
    adapter = OllamaAdapter({"model_name": "ollama/qwen3:4b", "supports_function_call": True})
    request = {
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
    }
    response = {
        "message": {
            "role": "assistant",
            "content": "I will call function get_weather(location='Paris')",
        }
    }

    with caplog.at_level(logging.WARNING, logger="model_calling.ollama_adapter"):
        adapter._log_request_response(request, response, is_tool_call=True)

    assert "may be attempting to call functions in text" in caplog.text
