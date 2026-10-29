"""
Client library for model-calling service.
"""

import json
import asyncio
import httpx
from typing import Dict, List, Any, Optional, Union, Generator


class ModelCallingClient:
    """Client for the model-calling service"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize with the service base URL"""
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)  # Increase timeout for capability testing

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 1.0,
        stream: bool = False,
        **kwargs,
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """Create a chat completion"""
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            **kwargs,
        }

        if tools:
            payload["tools"] = tools

        if stream:
            return self._stream_response(url, payload)
        else:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    async def embeddings(
        self, model: str, input: Union[str, List[str]], dimensions: Optional[int] = None, **kwargs
    ) -> Dict[str, Any]:
        """Create embeddings for the provided text"""
        url = f"{self.base_url}/v1/embeddings"

        payload = {"model": model, "input": input, **kwargs}

        if dimensions:
            payload["dimensions"] = dimensions

        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    async def _stream_response(
        self, url: str, payload: Dict[str, Any]
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream response chunks"""
        async with self.client.stream("POST", url, json=payload) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                if line.startswith("data: "):
                    line = line[6:]  # Remove "data: " prefix

                if line == "[DONE]":
                    break

                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue

    async def list_models(self) -> Dict[str, Any]:
        """List available models"""
        response = await self.client.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return response.json()

    async def update_config(self, prefix: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration for a model type"""
        payload = {"prefix": prefix, "config": config}
        response = await self.client.post(f"{self.base_url}/v1/config/update", json=payload)
        response.raise_for_status()
        return response.json()


# Synchronous wrapper for convenience
class SyncModelCallingClient:
    """Synchronous client for the model-calling service"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize with the service base URL"""
        self.async_client = ModelCallingClient(base_url)
        # A loop of its own: asyncio.get_event_loop() raises when no loop is
        # set, as in any thread but the main one, and in the main thread
        # too on Python 3.14.
        self.loop = asyncio.new_event_loop()

    def close(self):
        """Close the HTTP client and the client's event loop"""
        try:
            self.loop.run_until_complete(self.async_client.close())
        finally:
            self.loop.close()

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 1.0,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a chat completion"""
        return self.loop.run_until_complete(
            self.async_client.chat_completion(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                stream=stream,
                **kwargs,
            )
        )

    def embeddings(
        self, model: str, input: Union[str, List[str]], dimensions: Optional[int] = None, **kwargs
    ) -> Dict[str, Any]:
        """Create embeddings for the provided text"""
        return self.loop.run_until_complete(
            self.async_client.embeddings(model=model, input=input, dimensions=dimensions, **kwargs)
        )

    def list_models(self) -> Dict[str, Any]:
        """List available models"""
        return self.loop.run_until_complete(self.async_client.list_models())

    def update_config(self, prefix: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration for a model type"""
        return self.loop.run_until_complete(self.async_client.update_config(prefix, config))
