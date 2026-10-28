"""
Backends behind the provider interface in `providers.base`.

Anthropic and Cohere still go through the legacy adapters in
`model_calling.registry` until Phase 6 ports them.
"""

import os
from collections.abc import Mapping

from model_calling.providers.base import HttpClients, Provider, StreamParser
from model_calling.providers.ollama import OllamaProvider
from model_calling.providers.openai_compat import (
    OpenAICompatProvider,
    OpenAIProvider,
    OpenRouterProvider,
    VLLMProvider,
)

# Model-name prefix -> provider. Unprefixed names default to Ollama.
PROVIDERS: dict[str, type[Provider]] = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "openrouter": OpenRouterProvider,
    "vllm": VLLMProvider,
}
DEFAULT_PREFIX = "ollama"


def default_configs(environ: Mapping[str, str] = os.environ) -> dict[str, dict]:
    """Settings for every ported prefix, as `provider_for` reads them.

    Local backends get their usual local URLs. Hosted backends take their
    keys from the environment; nothing here reads a `.env` file.
    """
    org = environ.get("OPENAI_ORG_ID")
    return {
        "ollama": {"base_url": "http://localhost:11434"},
        # OpenAI-compatible base URLs are the API root, /v1 included.
        "vllm": {"base_url": "http://localhost:8000/v1"},
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key": environ.get("OPENAI_API_KEY", ""),
            "headers": {"OpenAI-Organization": org} if org else {},
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": environ.get("OPENROUTER_API_KEY", ""),
        },
    }


def split_model(model_name: str) -> tuple[str, str]:
    """`prefix/name` to (prefix, name). Unprefixed names default to Ollama."""
    if "/" in model_name:
        prefix, short_name = model_name.split("/", 1)
        return prefix, short_name
    return DEFAULT_PREFIX, model_name


def provider_for(
    model_name: str, configs: Mapping[str, dict], clients: HttpClients | None = None
) -> Provider | None:
    """The provider for `model_name`, or None if its prefix isn't ported yet.

    `configs` maps each prefix to its settings (`base_url`, `api_key`,
    `callbacks`, and optionally `headers` and `transport`), as in
    `ModelRegistry.default_configs`. With `clients`, the provider shares its
    prefix's pooled HTTP client instead of opening one per call.
    """
    prefix, short_name = split_model(model_name)
    cls = PROVIDERS.get(prefix)
    if cls is None:
        return None
    config = configs.get(prefix, {})
    if not config.get("base_url"):
        raise ValueError(f"no base_url configured for {prefix!r} models")
    return cls(
        model=short_name,
        base_url=config["base_url"],
        api_key=config.get("api_key") or None,
        callbacks=config.get("callbacks", ()),
        http_client=clients.get(prefix, config) if clients is not None else None,
        extra_headers=config.get("headers"),
    )


__all__ = [
    "HttpClients",
    "OllamaProvider",
    "OpenAICompatProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "PROVIDERS",
    "Provider",
    "StreamParser",
    "VLLMProvider",
    "default_configs",
    "provider_for",
    "split_model",
]
