"""
Base adapter classes for text embedding models.
"""

import time
import logging
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

from model_calling.callbacks import Callback

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class EmbeddingAdapter(ABC):
    """Base class for embedding model adapters"""

    def __init__(self, model_config: Dict[str, Any]):
        """Initialize the adapter with model-specific configuration."""
        self.model_config = model_config
        self.base_url = model_config.get("base_url")
        self.model_name = model_config.get("model_name")
        self.dimensions = model_config.get("dimensions", 1024)  # Default dimensions
        self.callbacks: List[Callback] = model_config.get("callbacks", [])

    @abstractmethod
    async def translate_request(self, openai_format_request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI format embedding request to model-specific format"""
        pass

    @abstractmethod
    async def translate_response(
        self, model_response: Dict[str, Any], input_texts: List[str]
    ) -> Dict[str, Any]:
        """Convert model-specific embedding response to OpenAI format"""
        pass

    @abstractmethod
    async def _call_model_api(self, translated_request: Dict[str, Any]) -> Dict[str, Any]:
        """Send the request to the model backend and return the raw response.

        This is the core API call that each adapter must implement. It should:
        1. Make the API call to their backend
        2. Handle any model-specific error cases
        3. Return the raw response

        The base class handles:
        - Callback notifications
        - Generic error handling
        - Response translation
        """
        pass

    async def call_model(self, translated_request: Dict[str, Any]) -> Dict[str, Any]:
        """High-level method to call the model with callback support."""
        input_texts = translated_request.get("input", [])

        # Call all callbacks before the request
        # Filter out 'model' key to avoid conflicts with positional parameter
        callback_kwargs = {k: v for k, v in translated_request.items() if k != "model"}
        for callback in self.callbacks:
            await callback.on_embedding_request_start(
                self.model_name, input_texts, **callback_kwargs
            )

        try:
            # Make the actual API call
            response = await self._call_model_api(translated_request)

            # Call all callbacks after the request
            for callback in self.callbacks:
                await callback.on_embedding_request_end(
                    self.model_name, input_texts, response, **callback_kwargs
                )

            return response
        except Exception as e:
            # Call error callbacks
            for callback in self.callbacks:
                await callback.on_error(self.model_name, input_texts, e, **callback_kwargs)
            raise


def create_openai_embedding_response(
    model: str,
    embeddings: List[List[float]],
    input_texts: List[str],
    dimensions: int = 1024,
    usage_tokens: int = 0,
) -> Dict[str, Any]:
    """Helper to create a standardized OpenAI-format embedding response"""

    # Ensure input_texts is a list
    if isinstance(input_texts, str):
        input_texts = [input_texts]

    logger.debug(
        f"Creating OpenAI embedding response for model: {model}, with {len(embeddings)} embeddings"
    )

    # Validate embeddings
    if not embeddings or not all(isinstance(emb, list) for emb in embeddings):
        logger.warning(f"Invalid embeddings format: {embeddings[:10] if embeddings else 'None'}")

    data = []
    for i, embedding in enumerate(embeddings):
        # Check if we have valid dimensions
        if not embedding or not isinstance(embedding, list):
            logger.warning(
                f"Invalid embedding at index {i}: {embedding[:5] if embedding else 'None'}"
            )
            # Create a fallback 0-vector
            embedding = [0.0] * dimensions

        data.append({"object": "embedding", "embedding": embedding, "index": i})

    # If we don't have token count, estimate based on text length
    if not usage_tokens:
        # Rough estimate: ~4 chars per token
        usage_tokens = sum(len(text) for text in input_texts) // 4

    response = {
        "object": "list",
        "data": data,
        "model": model,
        "usage": {"prompt_tokens": usage_tokens, "total_tokens": usage_tokens},
    }

    logger.debug(
        f"Created response with {len(data)} embeddings of dimensionality {len(embeddings[0]) if embeddings and embeddings[0] else 0}"
    )
    return response
