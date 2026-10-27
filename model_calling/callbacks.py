"""
Callback system for model-calling.

This module provides a callback interface that allows optional integration with external
services like tracers, loggers, etc. without tight coupling.
"""

from typing import Dict, List, Any, Optional


class Callback:
    """Base interface for callbacks to hook into model requests and responses"""

    async def on_request_start(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> None:
        """Called before sending a request to the model

        Args:
            model: Name of the model being called
            messages: List of messages in the request
            **kwargs: Additional request parameters
        """
        pass

    async def on_request_end(
        self, model: str, messages: List[Dict[str, Any]], response: Dict[str, Any], **kwargs
    ) -> None:
        """Called after receiving a response from the model

        Args:
            model: Name of the model being called
            messages: List of messages in the request
            response: The model's response
            **kwargs: Additional request parameters
        """
        pass

    async def on_stream_chunk(
        self, model: str, messages: List[Dict[str, Any]], chunk: Dict[str, Any], **kwargs
    ) -> None:
        """Called for each chunk in a streaming response

        Args:
            model: Name of the model being called
            messages: List of messages in the request
            chunk: The streaming chunk from the model
            **kwargs: Additional request parameters
        """
        pass

    async def on_error(
        self, model: str, messages: List[Dict[str, Any]], error: Exception, **kwargs
    ) -> None:
        """Called when an error occurs during the request

        Args:
            model: Name of the model being called
            messages: List of messages in the request
            error: The exception that occurred
            **kwargs: Additional request parameters
        """
        pass

    async def on_embedding_request_start(
        self, model: str, input_texts: List[str], **kwargs
    ) -> None:
        """Called before sending an embedding request to the model

        Args:
            model: Name of the model being called
            input_texts: List of input texts for embedding
            **kwargs: Additional request parameters
        """
        pass

    async def on_embedding_request_end(
        self, model: str, input_texts: List[str], response: Dict[str, Any], **kwargs
    ) -> None:
        """Called after receiving an embedding response from the model

        Args:
            model: Name of the model being called
            input_texts: List of input texts for embedding
            response: The model's response
            **kwargs: Additional request parameters
        """
        pass


class DefaultCallback(Callback):
    """Default callback implementation with basic logging"""

    def __init__(self, log_level: str = "INFO"):
        import logging

        self.logger = logging.getLogger(f"{__name__}.DefaultCallback")
        self.logger.setLevel(getattr(logging, log_level.upper()))

    async def on_request_start(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> None:
        self.logger.debug(f"Starting request to model: {model}")

    async def on_request_end(
        self, model: str, messages: List[Dict[str, Any]], response: Dict[str, Any], **kwargs
    ) -> None:
        self.logger.debug(f"Completed request to model: {model}")

    async def on_stream_chunk(
        self, model: str, messages: List[Dict[str, Any]], chunk: Dict[str, Any], **kwargs
    ) -> None:
        self.logger.debug(f"Received streaming chunk from model: {model}")

    async def on_error(
        self, model: str, messages: List[Dict[str, Any]], error: Exception, **kwargs
    ) -> None:
        self.logger.error(f"Error in request to model {model}: {error}")

    async def on_embedding_request_start(
        self, model: str, input_texts: List[str], **kwargs
    ) -> None:
        """Called before sending an embedding request to the model"""
        self.logger.debug(
            f"Starting embedding request to model: {model} for {len(input_texts)} inputs"
        )

    async def on_embedding_request_end(
        self, model: str, input_texts: List[str], response: Dict[str, Any], **kwargs
    ) -> None:
        """Called after receiving an embedding response from the model"""
        self.logger.debug(f"Completed embedding request to model: {model}")
