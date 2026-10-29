"""
Main entry point for the model-calling gateway: `python -m model_calling`
or the `model-calling` script. The gateway needs the `server` extra.
"""

import logging
import os
import sys

from model_calling.config import load_config

SERVER_EXTRA_HINT = (
    "The model-calling gateway needs the server extra "
    "({missing} is not installed). Install it with:\n"
    "    pip install 'model-calling[server]'"
)

logger = logging.getLogger(__name__)


def main():
    """Start the model-calling gateway"""
    try:
        import uvicorn
        from dotenv import find_dotenv, load_dotenv

        # Only the entry point reads .env, from the directory the gateway
        # starts in, and before the server builds its backend configs.
        # Importing model_calling.server never does.
        load_dotenv(find_dotenv(usecwd=True))
        import model_calling.server  # noqa: F401 - fail here, not inside uvicorn
    except ModuleNotFoundError as e:
        if e.name not in ("uvicorn", "fastapi", "dotenv", "starlette"):
            raise
        sys.exit(SERVER_EXTRA_HINT.format(missing=e.name))

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # Load configuration from environment
    config = load_config()

    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))

    # Log startup information
    logger.info(f"Starting model-calling service on port {port}")
    logger.info(f"Ollama base URL: {config['ollama']['base_url']}")
    logger.info(f"vLLM base URL: {config['vllm']['base_url']}")

    # Start the server
    uvicorn.run("model_calling.server:app", host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
