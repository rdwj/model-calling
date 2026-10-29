"""
Gateway backend settings from the environment.

The library reads only API keys from the environment
(`providers.default_configs`); the gateway also takes backend URLs from it,
so a container can point it at its backends.
"""

import os
from typing import Dict, Mapping, Optional, Tuple

# Enables POST /v1/config/update, which then wants it as a bearer token.
ADMIN_TOKEN_ENV = "MODEL_CALLING_ADMIN_TOKEN"

# Variable -> (model prefix, provider setting). URLs mean what `base_url`
# means everywhere: the API root with /v1 included, except Ollama's native
# API, which has none. The Anthropic variable is not `ANTHROPIC_BASE_URL`:
# the `anthropic` SDK and Claude Code read that name as the root without
# /v1, and a client pointed at this gateway sets it to the gateway itself.
BACKEND_ENV: Dict[str, Tuple[str, str]] = {
    "OLLAMA_BASE_URL": ("ollama", "base_url"),
    "VLLM_BASE_URL": ("vllm", "base_url"),
    "VLLM_API_KEY": ("vllm", "api_key"),
    "MODEL_CALLING_ANTHROPIC_BASE_URL": ("anthropic", "base_url"),
}


def backend_overrides(environ: Mapping[str, str] = os.environ) -> Dict[str, Dict[str, str]]:
    """Per-prefix settings from the set, non-empty `BACKEND_ENV` variables,
    in the shape `Client(config=...)` takes."""
    overrides: Dict[str, Dict[str, str]] = {}
    for var, (prefix, setting) in BACKEND_ENV.items():
        value = environ.get(var, "").strip()
        if not value:
            continue
        if setting == "base_url":
            value = value.rstrip("/")
        overrides.setdefault(prefix, {})[setting] = value
    return overrides


def admin_token(environ: Mapping[str, str] = os.environ) -> Optional[str]:
    """The token `/v1/config/update` requires, or None when it is disabled."""
    return environ.get(ADMIN_TOKEN_ENV, "").strip() or None
