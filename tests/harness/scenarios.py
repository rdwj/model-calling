"""
Scenarios and backends shared by the offline grid and the fixture recorder.

A scenario is one client request, written in Chat Completions form. A
backend knows how to phrase that scenario as its native request (for the
recorder) and how the gateway reaches it (for replay). Fixtures are stored
per (backend, scenario), so every ingress format replays the same backend
exchange.
"""

from dataclasses import dataclass, field
from typing import Tuple

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name"}},
            "required": ["location"],
        },
    },
}

TEXT_PROMPT = "Reply with exactly one word: pong"
TOOL_PROMPT = "Use the get_weather tool to look up the weather in Paris."


@dataclass(frozen=True)
class Scenario:
    name: str
    prompt: str
    stream: bool = False
    tools: Tuple[dict, ...] = ()
    max_tokens: int = 64
    # When set, the backend is asked for a model that doesn't exist.
    missing_model: bool = False

    def messages(self) -> list:
        return [{"role": "user", "content": self.prompt}]

    def chat_completions_body(self, model: str) -> dict:
        body = {
            "model": model,
            "messages": self.messages(),
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }
        if self.tools:
            body["tools"] = list(self.tools)
        return body


SCENARIOS = {
    s.name: s
    for s in (
        Scenario("text", TEXT_PROMPT),
        Scenario("tools", TOOL_PROMPT, tools=(WEATHER_TOOL,)),
        Scenario("stream", TEXT_PROMPT, stream=True),
        Scenario("backend_error", TEXT_PROMPT, missing_model=True),
    )
}


@dataclass(frozen=True)
class Backend:
    name: str
    # Model id a client sends to the gateway.
    gateway_model: str
    # Model id the backend itself expects, and one it doesn't have.
    native_model: str
    missing_native_model: str
    # Full URL of the backend's chat endpoint.
    endpoint: str
    # Registry prefix and the default_configs entries that point it here.
    config_prefix: str
    gateway_config: dict = field(default_factory=dict)
    # Fixture recording needs an OpenRouter key.
    needs_openrouter_key: bool = False

    def native_request(self, scenario: Scenario) -> dict:
        model = self.missing_native_model if scenario.missing_model else self.native_model
        if self.config_prefix == "ollama":
            # Mirrors OllamaAdapter.translate_request for these scenarios.
            body = {"model": model, "messages": scenario.messages(), "stream": scenario.stream}
            if scenario.tools:
                body["tools"] = list(scenario.tools)
            return body
        return scenario.chat_completions_body(model)


OLLAMA_URL = "http://localhost:11434"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
# Placeholder credential for replay. Real keys only ever come from the
# environment, in the recorder and live tests.
REPLAY_API_KEY = "replay-not-a-real-key"

BACKENDS = {
    b.name: b
    for b in (
        Backend(
            name="ollama",
            gateway_model="ollama/qwen3:4b",
            native_model="qwen3:4b",
            missing_native_model="model-calling-missing:latest",
            endpoint=f"{OLLAMA_URL}/api/chat",
            config_prefix="ollama",
            gateway_config={"base_url": OLLAMA_URL},
        ),
        Backend(
            name="openai_compat",
            gateway_model="openai/mistralai/mistral-nemo",
            native_model="mistralai/mistral-nemo",
            missing_native_model="model-calling/does-not-exist",
            endpoint=OPENROUTER_CHAT_URL,
            config_prefix="openai",
            gateway_config={"base_url": OPENROUTER_CHAT_URL, "api_key": REPLAY_API_KEY},
            needs_openrouter_key=True,
        ),
    )
}
