"""
Adapter for Ollama embedding models.
"""
import asyncio
import json
import logging
import sys
from typing import Dict, List, Any, Optional
import httpx

from model_calling.embedding_adapter import EmbeddingAdapter, create_openai_embedding_response

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class OllamaEmbeddingAdapter(EmbeddingAdapter):
    """Adapter for Ollama embedding models like nomic-embed-text"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.base_url = model_config.get("base_url", "http://localhost:11434")
        # Extract model name from the format "ollama/model-name:version"
        full_name = model_config.get("model_name", "")
        self.model_name = full_name.split("/")[-1] if "/" in full_name else full_name
        # Keep the version number
        
        # Set dimensions based on the model
        if "nomic-embed-text" in self.model_name:
            self.dimensions = 768  # Nomic Embed Text dimensions
        elif "mxbai-embed-large" in self.model_name:
            self.dimensions = 1024  # mxbai-embed-large dimensions
        else:
            # Default dimensions, can be overridden in model_config
            self.dimensions = model_config.get("dimensions", 1024)
        
        logger.info(f"Initialized Ollama embedding adapter for model {self.model_name} with {self.dimensions} dimensions")
    
    async def translate_request(self, openai_format_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI format embedding request to Ollama format"""
        # Extract input text
        input_text = openai_format_request.get("input", "")
        if isinstance(input_text, list):
            raise ValueError("Batch embeddings are not supported. Please use one text at a time.")
        
        logger.debug("Processing single text request")
        logger.debug(f"Input text (first 100 chars): {input_text[:100]}...")
        return {"model": self.model_name, "prompt": input_text}
    
    async def translate_response(self, ollama_response: Dict[str, Any], input_texts: List[str]) -> Dict[str, Any]:
        """Convert Ollama embedding response to OpenAI format"""
        # Extract embedding from response
        embedding = ollama_response.get("embedding", [])
        if not embedding:
            logger.error(f"No embedding found in response: {ollama_response}")
            embedding = [0.0] * self.dimensions
        
        return create_openai_embedding_response(
            embeddings=[embedding],
            model=self.model_name,
            input_texts=input_texts
        )
    
    async def _call_model_api(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Call Ollama API to get embeddings for a single text"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json=request
                )
                response.raise_for_status()
                result = response.json()
                if "error" in result:
                    raise Exception(result["error"])
                return result
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            # Only catch network/API errors, let validation errors propagate
            logger.error(f"API error in _call_model_api: {str(e)}")
            return {"embedding": [0.0] * self.dimensions}
        except Exception as e:
            # Re-raise other errors
            logger.error(f"Unexpected error in _call_model_api: {str(e)}")
            raise

    async def call_model(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """High-level method to call the model with callback support"""
        return await super().call_model(request)
