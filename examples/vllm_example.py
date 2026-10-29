"""
Sample usage of model-calling service with vLLM.
"""
import asyncio
from model_calling import Client


async def vllm_example():
    """Example with vLLM-deployed model"""
    # The library reaches vLLM at localhost:8000/v1 by default.
    # Override with config={"vllm": {"base_url": "...", "api_key": "..."}}
    async with Client() as client:
        # Basic completion with vLLM-deployed model
        response = await client.acompletion(
            model="vllm/mistral-7b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum computing in simple terms."}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        print("vLLM Response:")
        print(response.choices[0].message.content)

        # Streaming example with vLLM
        print("\nStreaming vLLM Response:")
        stream = await client.acompletion(
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
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content


if __name__ == "__main__":
    asyncio.run(vllm_example())
