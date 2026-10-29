"""
Main entry point for the model-calling gateway: `python -m model_calling`
or the `model-calling` script. The gateway needs the `server` extra.
"""

import logging
import os
import sys

SERVER_EXTRA_HINT = (
    "The model-calling gateway needs the server extra "
    "({missing} is not installed). Install it with:\n"
    "    pip install 'model-calling[server]'"
)

# Backends whose URL usually comes from the environment.
LOGGED_BACKENDS = ("ollama", "vllm", "anthropic")

logger = logging.getLogger(__name__)


def main():
    """Start the model-calling gateway"""
    try:
        import uvicorn
        from dotenv import load_dotenv

        # Only the entry point reads .env, from the directory the gateway
        # starts in (not its parents), and before the server builds its
        # backend configs. Importing model_calling.server never does.
        load_dotenv(os.path.join(os.getcwd(), ".env"))
        import model_calling.server  # fail here, not inside uvicorn
    except ModuleNotFoundError as e:
        if e.name not in ("uvicorn", "fastapi", "dotenv", "starlette"):
            raise
        sys.exit(SERVER_EXTRA_HINT.format(missing=e.name))

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))

    # The URLs the server will call, environment overrides applied.
    config = model_calling.server.registry.client.config
    logger.info(f"Starting model-calling service on port {port}")
    for prefix in LOGGED_BACKENDS:
        logger.info(f"{prefix} base URL: {config[prefix]['base_url']}")

    # Start the server
    uvicorn.run("model_calling.server:app", host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
