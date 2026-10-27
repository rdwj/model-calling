"""
Main entry point for the model-calling service.
"""
import uvicorn
import logging
import os
from model_calling.config import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def main():
    """Start the model-calling service"""
    # Load configuration from environment
    config = load_config()
    
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Log startup information
    logger.info(f"Starting model-calling service on port {port}")
    logger.info(f"Ollama base URL: {config['ollama']['base_url']}")
    logger.info(f"vLLM base URL: {config['vllm']['base_url']}")
    
    # Start the server
    uvicorn.run(
        "model_calling.server:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()
