"""
Adapter for Cohere embedding models.
"""
import json
import logging
import os
from typing import Dict, List, Any, Optional
import httpx

from model_calling.embedding_adapter import EmbeddingAdapter, create_openai_embedding_response

logger = logging.getLogger(__name__)


class CohereEmbeddingAdapter(EmbeddingAdapter):
    """Adapter for Cohere embedding models like embed-english-v3.0"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.base_url = model_config.get("base_url", "https://api.cohere.ai/v1/embed")
        # Extract model name from the format "cohere/model-name"
        full_name = model_config.get("model_name", "")
        self.model_name = full_name.split("/")[-1] if "/" in full_name else full_name
        
        # Get API key
        self.api_key = model_config.get("api_key", os.environ.get("COHERE_API_KEY", ""))
        if not self.api_key:
            logger.warning("No Cohere API key provided for embedding model")
        
        # Set dimensions based on the model
        model_dimensions = {
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-light-v3.0": 384,
            "embed-english-v2.0": 4096,
            "embed-english-light-v2.0": 1024,
            "embed-multilingual-v2.0": 768
        }
        
        self.dimensions = model_config.get(
            "dimensions", 
            model_dimensions.get(self.model_name, 1024)  # Default to 1024 if model unknown
        )
        
        logger.info(f"Initialized Cohere embedding adapter for model {self.model_name} with {self.dimensions} dimensions")
    
    async def translate_request(self, openai_format_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI format request to Cohere format"""
        # Cohere accepts both string and array of inputs
        input_text = openai_format_request.get("input", "")
        
        # Basic request structure for Cohere
        # Cohere uses 'texts' instead of 'input'
        cohere_request = {
            "model": self.model_name,
            "texts": input_text if isinstance(input_text, list) else [input_text],
            "input_type": "search_document"  # Default to document embedding
        }
        
        # If query specified, use query input type
        if openai_format_request.get("input_type") == "query" or "query" in str(input_text).lower():
            cohere_request["input_type"] = "search_query"
        
        # Add truncate parameter
        cohere_request["truncate"] = "END"  # Default behavior
        
        return cohere_request
    
    async def translate_response(self, cohere_response: Dict[str, Any], input_texts: List[str]) -> Dict[str, Any]:
        """Convert Cohere embedding response to OpenAI format"""
        # Extract the embeddings from Cohere response
        embeddings = cohere_response.get("embeddings", [])
        
        if not embeddings:
            logger.error(f"No embeddings found in Cohere response: {cohere_response}")
            # Return empty embeddings as fallback
            embeddings = [[0.0] * self.dimensions for _ in range(len(input_texts))]
        
        # Cohere doesn't provide token counting, so estimate based on text length
        # Rough estimate: ~4 chars per token
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        token_usage = sum(len(text) for text in input_texts) // 4
        
        # Create OpenAI-compatible response
        return create_openai_embedding_response(
            model=f"cohere/{self.model_name}",
            embeddings=embeddings,
            input_texts=input_texts,
            dimensions=self.dimensions,
            usage_tokens=token_usage
        )
    
    async def _call_model_api(self, translated_request: Dict[str, Any]) -> Dict[str, Any]:
        """Call the Cohere API and return the raw response"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        logger.debug(f"Sending embedding request to Cohere: {self.base_url}")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.base_url, 
                    json=translated_request, 
                    headers=headers,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.error(f"Error calling Cohere embeddings API: {str(e)}")
                raise
