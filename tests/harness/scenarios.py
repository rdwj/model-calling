"""
Scenarios and backends shared by the offline grid and the fixture recorder.

A scenario is one client request, written in Chat Completions form. A
backend knows how to phrase that scenario as its native request (for the
recorder) and how the gateway reaches it (for replay). Fixtures are stored
per (backend, scenario), so every ingress format replays the same backend
exchange.
"""

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping, Optional, Tuple

from model_calling.formats import anthropic, chat_completions, embeddings
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
PARALLEL_TOOL_PROMPT = (
    "Use the get_weather tool to look up the weather in Paris and in Tokyo. "
    "Call it once for each city, in the same turn."
)
IMAGE_PROMPT = "What color is this image? Reply with one word."
THINKING_PROMPT = "What is 17 * 23? Reply with just the number."
SYSTEM_BLOCKS = ("You are a terse assistant.", "Always answer in lowercase.")
STOP_SEQUENCE = "###"
# The tool call the tool_error scenario's history made, and its failed result.
PRIOR_CALL_ID = "call_weather_1"
PRIOR_CALL_INPUT = {"location": "Paris"}
TOOL_ERROR_TEXT = "Weather service unavailable: upstream timeout."

# A 64x64 solid red PNG.
IMAGE_PNG = (Path(__file__).parent / "red-square.png").read_bytes()
IMAGE_B64 = base64.b64encode(IMAGE_PNG).decode("ascii")

# Backend capabilities a scenario can need beyond text and tools.
VISION = "vision"
REASONING = "reasoning"
CHAT_COMPLETIONS = "chat_completions"
MESSAGES = "messages"


@dataclass(frozen=True)
class Scenario:
    """One client request, in both wire forms where both can express it.

    By default the request is the single user turn `prompt`. A scenario with
    history, system blocks, or images spells out each form's `messages` (and
    `messages_extra["system"]`) instead. `native_form` names the form the
    backend's native body is built from; strict replay then proves the other
    form produces the same body.
    """

    name: str
    prompt: str = ""
    stream: bool = False
    tools: Tuple[dict, ...] = ()
    # The same tools as Anthropic Messages tools.
    anthropic_tools: Tuple[dict, ...] = ()
    # Room for a reasoning model (gpt-oss) to think before it answers.
    max_tokens: int = 256
    # When set, the backend is asked for a model that doesn't exist.
    missing_model: bool = False
    # Explicit turns, overriding `prompt`, per wire form.
    chat_messages: Tuple[dict, ...] = ()
    anthropic_messages: Tuple[dict, ...] = ()
    # Further top-level fields per wire form.
    chat_extra: Mapping[str, Any] = field(default_factory=dict)
    messages_extra: Mapping[str, Any] = field(default_factory=dict)
    native_form: str = CHAT_COMPLETIONS
    # False when Chat Completions has no faithful form of the request.
    chat_form: bool = True
    # A capability the backend's model needs (VISION, REASONING), if any.
    needs: Optional[str] = None
    # Replay another scenario's fixture: the native exchange is identical,
    # and strict body matching proves it.
    fixture: Optional[str] = None

    def messages(self) -> list:
        return [{"role": "user", "content": self.prompt}]

    def chat_completions_body(self, model: str) -> dict:
        if not self.chat_form:
            raise ValueError(f"scenario {self.name!r} has no Chat Completions form")
        body = {
            "model": model,
            "messages": list(self.chat_messages) or self.messages(),
            "max_tokens": self.max_tokens,
            "stream": self.stream,
            **self.chat_extra,
        }
        if self.tools:
            body["tools"] = list(self.tools)
        return body

    def messages_body(self, model: str) -> dict:
        body = {
            "model": model,
            "messages": list(self.anthropic_messages) or self.messages(),
            "max_tokens": self.max_tokens,
            "stream": self.stream,
            **self.messages_extra,
        }
        if self.anthropic_tools:
            body["tools"] = list(self.anthropic_tools)
        return body

    def native_internal(self, model: str):
        """The scenario as an internal request, from its native form."""
        if self.native_form == MESSAGES:
            return anthropic.request_to_internal(self.messages_body(model))
        return chat_completions.request_to_internal(self.chat_completions_body(model))

    @property
    def fixture_name(self) -> str:
        return self.fixture or self.name


SCENARIOS = {
    s.name: s
    for s in (
        Scenario("text", TEXT_PROMPT),
        Scenario(
            "tools", TOOL_PROMPT, tools=(WEATHER_TOOL,), anthropic_tools=(ANTHROPIC_WEATHER_TOOL,)
        ),
        Scenario("stream", TEXT_PROMPT, stream=True),
        Scenario("backend_error", TEXT_PROMPT, missing_model=True),
        # System prompt as two blocks. The Messages form also carries fields
        # every backend here ignores: cache_control and metadata.
        Scenario(
            "system_blocks",
            chat_messages=(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": t} for t in SYSTEM_BLOCKS],
                },
                {"role": "user", "content": TEXT_PROMPT},
            ),
            anthropic_messages=({"role": "user", "content": TEXT_PROMPT},),
            chat_extra={"stop": [STOP_SEQUENCE]},
            messages_extra={
                "system": [
                    {"type": "text", "text": SYSTEM_BLOCKS[0]},
                    {
                        "type": "text",
                        "text": SYSTEM_BLOCKS[1],
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
                "stop_sequences": [STOP_SEQUENCE],
                "metadata": {"user_id": "grid-test"},
            },
        ),
        # A tool exchange whose result reports failure. Messages flags it
        # with is_error; Chat Completions has no such flag, and no backend
        # here takes one, so both forms send the same native body.
        Scenario(
            "tool_error",
            tools=(WEATHER_TOOL,),
            anthropic_tools=(ANTHROPIC_WEATHER_TOOL,),
            chat_messages=(
                {"role": "user", "content": TOOL_PROMPT},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": PRIOR_CALL_ID,
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps(PRIOR_CALL_INPUT),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": PRIOR_CALL_ID, "content": TOOL_ERROR_TEXT},
            ),
            anthropic_messages=(
                {"role": "user", "content": TOOL_PROMPT},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": PRIOR_CALL_ID,
                            "name": "get_weather",
                            "input": PRIOR_CALL_INPUT,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": PRIOR_CALL_ID,
                            "content": TOOL_ERROR_TEXT,
                            "is_error": True,
                        }
                    ],
                },
            ),
            native_form=MESSAGES,
        ),
        Scenario(
            "parallel_tools",
            PARALLEL_TOOL_PROMPT,
            # qwen3 thinks through both calls first.
            max_tokens=1024,
            tools=(WEATHER_TOOL,),
            anthropic_tools=(ANTHROPIC_WEATHER_TOOL,),
        ),
        Scenario(
            "image",
            chat_messages=(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{IMAGE_B64}"},
                        },
                        {"type": "text", "text": IMAGE_PROMPT},
                    ],
                },
            ),
            anthropic_messages=(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": IMAGE_B64,
                            },
                        },
                        {"type": "text", "text": IMAGE_PROMPT},
                    ],
                },
            ),
            needs=VISION,
        ),
        # Extended thinking with an effort level, which Chat Completions
        # backends take as reasoning_effort. Chat Completions' own
        # reasoning_effort is Phase 4's.
        Scenario(
            "thinking",
            THINKING_PROMPT,
            max_tokens=2048,
            messages_extra={
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "output_config": {"effort": "low"},
            },
            native_form=MESSAGES,
            chat_form=False,
            needs=REASONING,
        ),
        # The stream scenario with usage asked for. Messages streams always
        # carry usage, and backends are always asked for it, so the native
        # exchange is the stream fixture's.
        Scenario(
            "stream_usage",
            TEXT_PROMPT,
            stream=True,
            chat_extra={"stream_options": {"include_usage": True}},
            fixture="stream",
        ),
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
    # Capability -> the native model that has it, when `native_model`
    # doesn't. A capability missing here is assumed of `native_model`.
    capability_models: Mapping[str, str] = field(default_factory=dict)
    # Capability -> why this backend can't run scenarios needing it.
    lacks: Mapping[str, str] = field(default_factory=dict)
    # Scenarios recorded for this backend, if not all of them.
    only_scenarios: Optional[Tuple[str, ...]] = None

    def runs(self, scenario: Scenario) -> bool:
        return self.unsupported(scenario) is None and (
            self.only_scenarios is None or scenario.name in self.only_scenarios
        )

    def unsupported(self, scenario: Scenario) -> Optional[str]:
        """Why this backend can't run `scenario`, or None."""
        return self.lacks.get(scenario.needs) if scenario.needs else None

    def native_model_for(self, scenario: Scenario) -> str:
        if scenario.missing_model:
            return self.missing_native_model
        return self.capability_models.get(scenario.needs, self.native_model)

    def model_for(self, scenario: Scenario) -> str:
        """The model id a client sends the gateway for `scenario`."""
        return f"{self.config_prefix}/{self.native_model_for(scenario)}"

    def native_request(self, scenario: Scenario) -> dict:
        """The body the gateway sends this backend for `scenario`."""
        model = self.native_model_for(scenario)
        request = scenario.native_internal(model)
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
# Models for scenarios the default model can't run. qwen3:4b, gpt-oss-20b,
# and the OpenRouter reasoning model all think.
OLLAMA_VISION_MODEL = "gemma3"
OPENROUTER_VISION_MODEL = "google/gemma-3-4b-it"
OPENROUTER_REASONING_MODEL = "openai/gpt-oss-20b"
VLLM_NO_VISION = "vLLM serves gpt-oss-20b, which takes no image input"

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
            capability_models={VISION: OLLAMA_VISION_MODEL},
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
            capability_models={
                VISION: OPENROUTER_VISION_MODEL,
                REASONING: OPENROUTER_REASONING_MODEL,
            },
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
            lacks={VISION: VLLM_NO_VISION},
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
            lacks={VISION: VLLM_NO_VISION},
            only_scenarios=("text", "tools", "stream", "backend_error"),
        ),
    )
}

GRID_BACKENDS = tuple(name for name, b in BACKENDS.items() if b.in_grid)


def recorded_scenarios(backend: Backend) -> list:
    """The scenarios with a fixture of their own for `backend`."""
    return [s for s in SCENARIOS.values() if backend.runs(s) and s.fixture is None]


# --- embeddings ---------------------------------------------------------------

EMBEDDING_INPUTS = ("The sky is blue.", "Grass is green.")


@dataclass(frozen=True)
class EmbeddingScenario:
    name: str
    inputs: Tuple[str, ...] = EMBEDDING_INPUTS
    missing_model: bool = False
    # Embeddings never stream; the recorder reads this off any scenario.
    stream: ClassVar[bool] = False

    def body(self, model: str) -> dict:
        return {"model": model, "input": list(self.inputs)}


EMBEDDING_SCENARIOS = {
    s.name: s
    for s in (
        EmbeddingScenario("embedding"),
        EmbeddingScenario("embedding_error", missing_model=True),
    )
}

OLLAMA_EMBED_MODEL = "nomic-embed-text"


@dataclass(frozen=True)
class EmbeddingBackend:
    """An embeddings endpoint, in the same terms `Backend` uses for chat, so
    the recorder and the `library`/`gateway` fixtures can treat both alike."""

    name: str
    gateway_model: str
    native_model: str
    missing_native_model: str
    endpoint: str
    config_prefix: str
    provider: type[Provider]
    gateway_config: dict = field(default_factory=dict)
    needs_openrouter_key: bool = False

    def native_model_for(self, scenario: EmbeddingScenario) -> str:
        return self.missing_native_model if scenario.missing_model else self.native_model

    def model_for(self, scenario: EmbeddingScenario) -> str:
        return f"{self.config_prefix}/{self.native_model_for(scenario)}"

    def native_request(self, scenario: EmbeddingScenario) -> dict:
        model = self.native_model_for(scenario)
        request = embeddings.request_to_internal(scenario.body(model))
        provider = self.provider(model=model, base_url=self.gateway_config["base_url"])
        return provider.build_embedding_request(request)


EMBEDDING_BACKENDS = {
    b.name: b
    for b in (
        EmbeddingBackend(
            name="ollama_embed",
            gateway_model=f"ollama/{OLLAMA_EMBED_MODEL}",
            native_model=OLLAMA_EMBED_MODEL,
            missing_native_model="model-calling-missing:latest",
            endpoint=f"{OLLAMA_URL}/api/embed",
            config_prefix="ollama",
            gateway_config={"base_url": OLLAMA_URL},
            provider=OllamaProvider,
        ),
        # Ollama's OpenAI-compatible root, standing in for vLLM: the same
        # /v1/embeddings wire format, recordable for free. See fixtures/README.md.
        EmbeddingBackend(
            name="openai_compat_embed",
            gateway_model=f"vllm/{OLLAMA_EMBED_MODEL}",
            native_model=OLLAMA_EMBED_MODEL,
            missing_native_model="model-calling-missing:latest",
            endpoint=f"{OLLAMA_URL}/v1/embeddings",
            config_prefix="vllm",
            gateway_config={"base_url": f"{OLLAMA_URL}/v1", "api_key": REPLAY_API_KEY},
            provider=VLLMProvider,
        ),
    )
}

# Every recorded backend and scenario, chat and embeddings.
ALL_BACKENDS = {**BACKENDS, **EMBEDDING_BACKENDS}
ALL_SCENARIOS = {**SCENARIOS, **EMBEDDING_SCENARIOS}


def fixture_for(backend_name: str, scenario_name: str) -> Path:
    """The fixture a (backend, scenario) cell replays."""
    from tests.harness.replay import fixture_path

    scenario = ALL_SCENARIOS[scenario_name]
    return fixture_path(backend_name, getattr(scenario, "fixture_name", scenario_name))
