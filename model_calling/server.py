"""
FastAPI server for the model-calling service.
"""

import contextlib
import hmac
import logging
import os
from typing import Any, List, Optional, Union
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

from model_calling import __version__
from model_calling import gateway
from model_calling.aliases import AliasStore
from model_calling.providers import PROVIDERS, split_model
from model_calling.registry import registry
from model_calling.config import admin_token


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await registry.client.aclose()


app = FastAPI(title="Model Calling Service", version=__version__, lifespan=lifespan)

logger = logging.getLogger(__name__)

NOT_IMPLEMENTED = 501
# Set to true for strict mode: a parameter the backend can't honor gets a
# 400 naming it. By default the gateway drops such parameters.
STRICT_PARAMS_ENV = "MODEL_CALLING_STRICT_PARAMS"
_BOOLEANS = {
    "true": True,
    "1": True,
    "yes": True,
    "false": False,
    "0": False,
    "no": False,
    "": False,
}


def strict_params(environ=os.environ) -> bool:
    value = environ.get(STRICT_PARAMS_ENV, "").strip().lower()
    if value not in _BOOLEANS:
        raise RuntimeError(f"{STRICT_PARAMS_ENV} must be true or false, not {value!r}")
    return _BOOLEANS[value]


DROP_PARAMS = not strict_params()
# The model aliases clients use, from the file MODEL_CALLING_CONFIG names.
MODELS = AliasStore.from_env()
# `owned_by` in the models list: the backend behind an alias isn't the client's concern.
MODEL_OWNER = "model-calling"


def _ported(body: Any) -> bool:
    """Whether the library `Client` serves this request's model prefix."""
    model = body.get("model") if isinstance(body, dict) else None
    return isinstance(model, str) and split_model(model)[0] in PROVIDERS


class ModerationRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None


@app.post("/v1/moderations")
async def moderations(request: Request):
    """OpenAI-compatible content moderation endpoint"""
    # Parse the request body
    body = await request.json()

    # Validate request format
    try:
        validated_request = ModerationRequest(**body)
    except Exception as e:
        logger.error(f"Invalid request format: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid request format: {e}")

    # Moderation always runs on Granite Guardian. Resolving the model through
    # the chat registry would send unprefixed names to Ollama's chat API.
    model_name = validated_request.model or "granite-guardian"
    # Imported lazily: torch and transformers are an optional extra
    try:
        from model_calling.granite_guardian_adapter import GraniteGuardianAdapter
    except ImportError as e:
        raise HTTPException(
            status_code=NOT_IMPLEMENTED,
            detail=f"Moderation requires optional dependencies ({e.name} is missing). "
            "Install with: pip install 'model-calling[moderation]'",
        )
    adapter = GraniteGuardianAdapter({"model_name": model_name})

    # Convert to model-specific format
    try:
        translated_request = await adapter.translate_request(body)
    except Exception as e:
        logger.error(f"Error translating request: {e}")
        raise HTTPException(status_code=500, detail=f"Error translating request: {e}")

    # Call the model
    try:
        model_response = await adapter.call_model(translated_request)
        translated_response = await adapter.translate_response(model_response)

        # Return the moderation results
        return translated_response
    except Exception as e:
        logger.error(f"Error calling model: {e}")
        raise HTTPException(status_code=500, detail=f"Error calling model: {e}")


@app.get("/")
async def root():
    """Service health check endpoint"""
    return {"status": "ok", "service": "model-calling"}


@app.get("/v1/models")
async def list_models():
    """The configured model aliases, as the OpenAI models list."""
    return {
        "object": "list",
        "data": [
            {"id": alias, "object": "model", "created": 0, "owned_by": MODEL_OWNER}
            for alias in MODELS.current().aliases
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI Chat Completions, for any configured model."""
    body = await request.json()
    return await gateway.chat_completions(
        registry.client, body, MODELS.current(), drop_params=DROP_PARAMS
    )


@app.post("/v1/messages")
async def messages(request: Request):
    """Anthropic Messages over every ported backend."""
    body = await request.json()
    return await gateway.messages(registry.client, body, MODELS.current(), drop_params=DROP_PARAMS)


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    """Anthropic token counting, as an estimate: no backend here counts tokens."""
    return gateway.count_tokens(await request.json(), MODELS.current())


@app.post("/v1/embeddings")
async def create_embeddings(request: Request):
    """OpenAI Embeddings, for any configured model."""
    body = await request.json()
    return await gateway.embeddings(
        registry.client, body, MODELS.current(), drop_params=DROP_PARAMS
    )


def require_admin(request: Request) -> None:
    """Gate an admin endpoint: absent (404) unless the admin token is set,
    then forbidden (403) without it as a bearer token."""
    token = admin_token()
    if token is None:
        raise HTTPException(status_code=404, detail="Not Found")
    scheme, _, credential = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(
        credential.strip().encode(), token.encode()
    ):
        raise HTTPException(status_code=403, detail="Admin token required")


@app.post("/v1/config/update")
async def update_config(request: Request):
    """Update configuration for a model type. Can redirect a backend, and
    the hosted key that goes with it, so it needs the admin token."""
    require_admin(request)
    body = await request.json()
    prefix = body.get("prefix")
    config = body.get("config", {})

    if not prefix:
        raise HTTPException(status_code=400, detail="Missing prefix parameter")

    # Update the registry configuration
    registry.update_config(prefix, config)

    return {"status": "ok", "message": f"Updated configuration for {prefix}"}


@app.post("/v1/models/refresh")
async def refresh_model(request: Request):
    """Force refresh of model capabilities"""
    body = await request.json()
    model_name = body.get("model")

    if not model_name:
        raise HTTPException(status_code=400, detail="Missing model parameter")

    # Clear the adapter cache to force re-verification
    registry.adapters = {}

    # Ported prefixes without a legacy adapter (vllm, openrouter) have no
    # capability probe; report the registry's capabilities for them.
    adapter = registry.get_adapter(model_name)
    if not adapter and _ported(body):
        adapter = registry.get_capabilities(model_name, ignore_cache=True)
    if not adapter:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    # Return the updated capabilities
    capabilities = {
        "supports_function_call": (
            adapter.supports_function_call if hasattr(adapter, "supports_function_call") else False
        ),
        "supports_streaming": (
            adapter.supports_streaming if hasattr(adapter, "supports_streaming") else False
        ),
        "supports_system_messages": (
            adapter.supports_system_messages
            if hasattr(adapter, "supports_system_messages")
            else False
        ),
        "supports_vision": (
            adapter.supports_vision if hasattr(adapter, "supports_vision") else False
        ),
        "supports_json_mode": (
            adapter.supports_json_mode if hasattr(adapter, "supports_json_mode") else False
        ),
        "supports_reasoning": (
            adapter.supports_reasoning if hasattr(adapter, "supports_reasoning") else False
        ),
    }

    return {"model": model_name, "capabilities": capabilities, "status": "refreshed"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
