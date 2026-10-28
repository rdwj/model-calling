"""
Backends behind the provider interface in `providers.base`.

Only Ollama has moved here so far; other prefixes still go through the
legacy adapters in `model_calling.registry` until they are ported.
"""

from collections.abc import Mapping

from model_calling.providers.base import BackendError, Provider, StreamParser
from model_calling.providers.ollama import OllamaProvider

# Model-name prefix -> provider. Unprefixed names default to Ollama.
PROVIDERS: dict[str, type[Provider]] = {"ollama": OllamaProvider}
DEFAULT_PREFIX = "ollama"


def provider_for(model_name: str, configs: Mapping[str, dict]) -> Provider | None:
    """The provider for `model_name`, or None if its prefix isn't ported yet.

    `configs` maps each prefix to its settings (`base_url`, `api_key`,
    `callbacks`), as in `ModelRegistry.default_configs`.
    """
    if "/" in model_name:
        prefix, short_name = model_name.split("/", 1)
    else:
        prefix, short_name = DEFAULT_PREFIX, model_name
    cls = PROVIDERS.get(prefix)
    if cls is None:
        return None
    config = configs.get(prefix, {})
    return cls(
        model=short_name,
        base_url=config["base_url"],
        api_key=config.get("api_key"),
        callbacks=config.get("callbacks", ()),
    )


__all__ = [
    "BackendError",
    "OllamaProvider",
    "PROVIDERS",
    "Provider",
    "StreamParser",
    "provider_for",
]
