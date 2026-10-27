"""
Adapter for OpenAI embedding models.
"""
import json
import logging
import os
from typing import Dict, List, Any, Optional
import httpx

from model_calling.embedding_adapter import EmbeddingAdapter, create_openai_embedding_response

logger = logging.getLogger(__name__)


class OpenAIEmbeddingAdapter(EmbeddingAdapter):
    """Adapter for OpenAI embedding models like text-embedding-ada-002"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.base_url = model_config.get("base_url", "https://api.openai.com/v1/embeddings")
        # Extract model name from the format "openai/model-name"
        full_name = model_config.get("model_name", "")
        self.model_name = full_name.split("/")[-1] if "/" in full_name else full_name
        
        # Get API key
        self.api_key = model_config.get("api_key", os.environ.get("OPENAI_API_KEY", ""))
        if not self.api_key:
            logger.warning("No OpenAI API key provided for embedding model")
        
        # Set dimensions based on the model
        model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        
        self.dimensions = model_config.get(
            "dimensions", 
            model_dimensions.get(self.model_name, 1536)  # Default to 1536 if model unknown
        )
        
        logger.info(f"Initialized OpenAI embedding adapter for model {self.model_name} with {self.dimensions} dimensions")
    
    async def translate_request(self, openai_format_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert our request to OpenAI format (mostly a pass-through)"""
        # OpenAI accepts an array of inputs
        input_text = openai_format_request.get("input", "")
        
        # Format properly - OpenAI expects input to be array for batch processing
        if isinstance(input_text, str):
            input_text = [input_text]
        
        request = {
            "model": self.model_name,
            "input": input_text
        }
        
        # Add optional parameters
        if "dimensions" in openai_format_request:
            request["dimensions"] = openai_format_request["dimensions"]
        
        if "user" in openai_format_request:
            request["user"] = openai_format_request["user"]
        
        return request
    
    async def translate_response(self, openai_response: Dict[str, Any], input_texts: List[str]) -> Dict[str, Any]:
        """Convert OpenAI embedding response to our standardized format (mostly a pass-through)"""
        # Extract token usage if available
        usage = openai_response.get("usage", {})
        token_usage = usage.get("total_tokens", 0)
        
        # Extract embeddings
        data = openai_response.get("data", [])
        embeddings = [item.get("embedding", []) for item in data]
        
        if not embeddings:
            logger.error(f"No embeddings found in OpenAI response: {openai_response}")
            # Return empty embeddings as fallback
            embeddings = [[0.0] * self.dimensions for _ in range(len(input_texts))]
        
        # Create standardized response
        return create_openai_embedding_response(
            model=f"openai/{self.model_name}",
            embeddings=embeddings,
            input_texts=input_texts,
            dimensions=self.dimensions,
            usage_tokens=token_usage
        )
    
    async def _call_model_api(self, translated_request: Dict[str, Any]) -> Dict[str, Any]:
        """Call the OpenAI API and return the raw response"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        logger.debug(f"Sending embedding request to OpenAI: {self.base_url}")
        
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
                logger.error(f"Error calling OpenAI embeddings API: {str(e)}")
                raise
