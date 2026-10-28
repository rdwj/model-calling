"""
Sample usage of model-calling service with vLLM.
"""
import asyncio
import json
from model_calling.client import ModelCallingClient


async def vllm_example():
    """Example with vLLM-deployed model"""
    client = ModelCallingClient()
    
    try:
        # First, update config if your vLLM instance is not on localhost:8000/v1
        await client.update_config(
            prefix="vllm",
            config={
                "base_url": "http://your-vllm-server:8000/v1",
            }
        )
        
        # Basic completion with vLLM-deployed model
        response = await client.chat_completion(
            model="vllm/mistral-7b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum computing in simple terms."}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        print("vLLM Response:")
        print(response["choices"][0]["message"]["content"])
        
        # Streaming example with vLLM
        print("\nStreaming vLLM Response:")
        stream = await client.chat_completion(
            model="vllm/mistral-7b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a short story about a robot learning to paint."}
            ],
            temperature=0.7,
            max_tokens=500,
            stream=True
        )
        
        full_response = ""
        async for chunk in stream:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta and delta["content"]:
                    content = delta["content"]
                    print(content, end="", flush=True)
                    full_response += content
    
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(vllm_example())
