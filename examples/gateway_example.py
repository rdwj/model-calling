"""
Example of using the model-calling gateway with official OpenAI and Anthropic SDKs.

Start the gateway with:
  podman-compose up

Then run this script.
"""
import asyncio
import os
import openai
import anthropic


async def test_openai_sdk():
    """Test the gateway with the official OpenAI SDK"""
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:8000")

    client = openai.AsyncOpenAI(
        base_url=f"{gateway_url}/v1",
        api_key="unused"
    )

    print("=== OpenAI SDK: Non-streaming ===")
    response = await client.chat.completions.create(
        model="chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )
    print(response.choices[0].message.content)

    print("\n=== OpenAI SDK: Streaming ===")
    stream = await client.chat.completions.create(
        model="chat",
        messages=[
            {"role": "user", "content": "Count to 5."}
        ],
        stream=True
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


async def test_anthropic_sdk():
    """Test the gateway with the official Anthropic SDK"""
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:8000")

    client = anthropic.AsyncAnthropic(
        base_url=gateway_url,
        api_key="unused"
    )

    print("=== Anthropic SDK: Non-streaming ===")
    response = await client.messages.create(
        model="chat",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "What is the capital of Germany?"}
        ]
    )
    # Thinking models (qwen3) send a thinking block before the text.
    print("".join(block.text for block in response.content if block.type == "text"))

    print("\n=== Anthropic SDK: Streaming ===")
    async with client.messages.stream(
        model="chat",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Count to 3."}
        ]
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)
    print("\n")


async def main():
    """Run all examples"""
    await test_openai_sdk()
    await test_anthropic_sdk()


if __name__ == "__main__":
    asyncio.run(main())
