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

from model_calling.formats import anthropic, chat_completions
from model_calling.providers import OllamaProvider, OpenRouterProvider, Provider, VLLMProvider

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

# WEATHER_TOOL as an Anthropic Messages tool.
ANTHROPIC_WEATHER_TOOL = {
    "name": WEATHER_TOOL["function"]["name"],
    "description": WEATHER_TOOL["function"]["description"],
    "input_schema": WEATHER_TOOL["function"]["parameters"],
}

TEXT_PROMPT = "Reply with exactly one word: pong"
TOOL_PROMPT = "Use the get_weather tool to look up the weather in Paris."


@dataclass(frozen=True)
class Scenario:
    name: str
    prompt: str
    stream: bool = False
    tools: Tuple[dict, ...] = ()
    # The same tools as Anthropic Messages tools.
    anthropic_tools: Tuple[dict, ...] = ()
    # Room for a reasoning model (gpt-oss) to think before it answers.
    max_tokens: int = 256
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

    def messages_body(self, model: str) -> dict:
        body = {
            "model": model,
            "messages": self.messages(),
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }
        if self.anthropic_tools:
            body["tools"] = list(self.anthropic_tools)
        return body


SCENARIOS = {
    s.name: s
    for s in (
        Scenario("text", TEXT_PROMPT),
        Scenario(
            "tools", TOOL_PROMPT, tools=(WEATHER_TOOL,), anthropic_tools=(ANTHROPIC_WEATHER_TOOL,)
        ),
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
    # The provider that builds this backend's native body. None for an
    # Anthropic Messages endpoint, whose body comes from formats.anthropic.
    provider: type[Provider] | None
    gateway_config: dict = field(default_factory=dict)
    # Fixture recording needs an OpenRouter key.
    needs_openrouter_key: bool = False
    # False for a backend recorded only as a reference, because no provider
    # can reach it yet. Its fixtures are validated but stay off the grid.
    in_grid: bool = True

    def model_for(self, scenario: Scenario) -> str:
        """The model id a client sends the gateway for `scenario`."""
        if scenario.missing_model:
            return f"{self.config_prefix}/{self.missing_native_model}"
        return self.gateway_model

    def native_request(self, scenario: Scenario) -> dict:
        """The body the gateway sends this backend for `scenario`."""
        model = self.missing_native_model if scenario.missing_model else self.native_model
        request = chat_completions.request_to_internal(scenario.chat_completions_body(model))
        if self.provider is None:
            return anthropic.request_from_internal(request)
        provider = self.provider(model=model, base_url=self.gateway_config["base_url"])
        return {**provider.build_request(request), "stream": scenario.stream}


OLLAMA_URL = "http://localhost:11434"
# The registry's default OpenRouter base URL.
OPENROUTER_URL = "https://openrouter.ai/api/v1"
# Where replay expects vLLM. Recording sends to VLLM_BASE_URL and VLLM_MODEL
# from the environment and stores these in their place.
VLLM_URL = "http://localhost:8000/v1"
VLLM_MODEL = "gpt-oss-20b"
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
            provider=OllamaProvider,
        ),
        Backend(
            name="openai_compat",
            gateway_model="openrouter/mistralai/mistral-nemo",
            native_model="mistralai/mistral-nemo",
            missing_native_model="model-calling/does-not-exist",
            endpoint=f"{OPENROUTER_URL}/chat/completions",
            config_prefix="openrouter",
            gateway_config={"base_url": OPENROUTER_URL, "api_key": REPLAY_API_KEY},
            needs_openrouter_key=True,
            provider=OpenRouterProvider,
        ),
        Backend(
            name="vllm",
            gateway_model=f"vllm/{VLLM_MODEL}",
            native_model=VLLM_MODEL,
            missing_native_model="model-calling-missing",
            endpoint=f"{VLLM_URL}/chat/completions",
            config_prefix="vllm",
            gateway_config={"base_url": VLLM_URL, "api_key": REPLAY_API_KEY},
            provider=VLLMProvider,
        ),
        # vLLM's native Anthropic Messages endpoint, the reference for our
        # Messages encoder until an Anthropic provider exists (Phase 6).
        Backend(
            name="vllm_messages",
            gateway_model=f"vllm/{VLLM_MODEL}",
            native_model=VLLM_MODEL,
            missing_native_model="model-calling-missing",
            endpoint=f"{VLLM_URL}/messages",
            config_prefix="vllm",
            gateway_config={"base_url": VLLM_URL, "api_key": REPLAY_API_KEY},
            provider=None,
            in_grid=False,
        ),
    )
}

GRID_BACKENDS = tuple(name for name, b in BACKENDS.items() if b.in_grid)
