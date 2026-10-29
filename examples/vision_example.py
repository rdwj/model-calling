"""
Example of using vision capabilities with model-calling service.
"""
import asyncio
import base64
import os
from model_calling import Client


async def generate_alt_text(image_path, prompt=None):
    """Generate alt text for an image using a vision model"""
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return None

    # Encode the image to base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Default prompt for alt text generation
    if prompt is None:
        prompt = (
            "Please generate a concise and descriptive alt text for this image. "
            "The alt text should be informative, objective, and suitable for "
            "accessibility purposes. Focus on the main elements visible in the "
            "image and their relationships. Keep it under 100 words."
        )

    # Create the client
    async with Client() as client:
        # Format the message with the image
        message_content = f"{prompt}\n\n![Image](data:image/png;base64,{base64_image})"

        try:
            # Call the vision model
            print(f"Calling vision model to analyze image: {os.path.basename(image_path)}")
            response = await client.acompletion(
                model="ollama/llava:7b",  # Using Llava vision model
                messages=[
                    {"role": "user", "content": message_content}
                ]
            )

            # Extract and return the generated alt text
            alt_text = response.choices[0].message.content
            print("\nGenerated Alt Text:")
            print(alt_text)
            return alt_text

        except Exception as e:
            print(f"Error generating alt text: {e}")
            return None


async def vision_classification_example(image_path):
    """Example of using vision model for image classification"""
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return None

    # Encode the image to base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Create the client
    async with Client() as client:
        # Format the message with the image and classification prompt
        message_content = (
            "Please classify what's in this image. Provide a short list of categories "
            "that best describe the content.\n\n"
            f"![Image](data:image/png;base64,{base64_image})"
        )

        try:
            # Call the vision model
            print(f"Classifying image: {os.path.basename(image_path)}")
            response = await client.acompletion(
                model="ollama/llava:7b",  # Using Llava vision model
                messages=[
                    {"role": "user", "content": message_content}
                ]
            )

            # Extract and return the classification
            classification = response.choices[0].message.content
            print("\nImage Classification:")
            print(classification)
            return classification

        except Exception as e:
            print(f"Error classifying image: {e}")
            return None


async def main():
    """Run all examples"""
    # Example image path - replace with your own image
    image_path = "../test_image.jpg"  # Default to the test image we created earlier
    
    # If the test image doesn't exist, try to find another image
    if not os.path.exists(image_path):
        # Try to find any image in the current directory
        for ext in ['.jpg', '.jpeg', '.png']:
            files = [f for f in os.listdir('.') if f.endswith(ext)]
            if files:
                image_path = files[0]
                break
    
    print("=" * 50)
    print("Vision Model Examples")
    print("=" * 50)
    
    print("\n1. Alt Text Generation Example")
    print("-" * 30)
    await generate_alt_text(image_path)
    
    print("\n" + "=" * 50 + "\n")
    
    print("2. Image Classification Example")
    print("-" * 30)
    await vision_classification_example(image_path)


if __name__ == "__main__":
    asyncio.run(main())
