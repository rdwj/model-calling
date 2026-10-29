"""
Example of using embedding models through the model-calling service.
"""
import asyncio
import numpy as np
from model_calling import Client

# Helper function for calculating cosine similarity
def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def test_embeddings():
    """Test embeddings with Ollama models"""
    async with Client() as client:
        print("\n=== Testing Embeddings ===")

        # Single text embedding
        text = "The quick brown fox jumps over the lazy dog."
        print(f"Embedding text: {text}")

        # Use the verified models with version numbers
        embedding_models = [
            "ollama/nomic-embed-text:v1.5",  # Verified working
            "ollama/mxbai-embed-large:335m"  # Verified working
        ]

        # Try each model until one works
        for model in embedding_models:
            try:
                print(f"\nTrying with model: {model}")
                response = await client.aembedding(
                    model=model,
                    input=text
                )

                # Get the embedding vector
                embedding = response.data[0].embedding
                dimensions = len(embedding)

                print(f"Success! Received embedding with {dimensions} dimensions")
                print(f"First 5 values: {embedding[:5]}")
                break  # Exit loop if successful
            except Exception as e:
                print(f"Error with {model}: {str(e)}")
                if model == embedding_models[-1]:  # If this was the last model to try
                    print("\nAll embedding models failed. Please check your Ollama installation.")
                    return  # Exit the function early
        
        # Compare multiple texts
        texts = [
            "I love machine learning and natural language processing.",
            "Deep learning models can process text effectively.",
            "The stock market is volatile today.",
            "Financial analysis shows the economy is improving."
        ]

        print("\nEmbedding multiple texts for similarity comparison:")
        for i, text in enumerate(texts):
            print(f"Text {i+1}: {text}")

        # Get embeddings for all texts
        embedding_models = [
            "ollama/nomic-embed-text:v1.5",  # Verified working
            "ollama/mxbai-embed-large:335m"  # Verified working
        ]

        # Try each model for batch processing
        for model in embedding_models:
            try:
                print(f"\nTrying batch embedding with model: {model}")
                response = await client.aembedding(
                    model=model,
                    input=texts
                )
                break  # Exit loop if successful
            except Exception as e:
                print(f"Error with {model} for batch processing: {str(e)}")
                if model == embedding_models[-1]:  # If this was the last model to try
                    print("\nAll embedding models failed for batch processing. Exiting.")
                    return  # Exit the function early

        # Extract embeddings
        embeddings = [item.embedding for item in response.data]
        print(f"\nNumber of embeddings: {len(embeddings)}")
        print(f"Number of texts: {len(texts)}")

        # Calculate similarity matrix
        print("\nSimilarity matrix:")
        similarity_matrix = np.zeros((len(texts), len(texts)))

        for i in range(len(texts)):
            for j in range(len(texts)):
                similarity = cosine_similarity(embeddings[i], embeddings[j])
                similarity_matrix[i][j] = similarity

        # Print similarity matrix with formatting
        print("      " + "       ".join([f"Text {i+1}" for i in range(len(texts))]))
        for i in range(len(texts)):
            row = [f"Text {i+1}"]
            for j in range(len(texts)):
                row.append(f"{similarity_matrix[i][j]:.5f}")
            print(" ".join(row))

        print("\nObservations:")
        print("- Text 1 and Text 2 are both about ML/NLP and have high similarity")
        print("- Text 3 and Text 4 are both about finance and have high similarity")
        print("- Texts about ML/NLP have low similarity with texts about finance")


async def test_embedding_search():
    """Test a simple embedding-based search example"""
    async with Client() as client:
        print("\n=== Testing Embedding Search ===")

        # Create a small document collection
        documents = [
            "Artificial intelligence is revolutionizing healthcare with advanced diagnostics.",
            "Machine learning algorithms can predict patient outcomes with increasing accuracy.",
            "The stock market showed significant volatility due to economic uncertainty.",
            "Financial analysts predict interest rates will remain stable through next quarter.",
            "Natural language processing is helping researchers analyze medical literature.",
            "Deep learning models can now detect patterns in medical images better than humans."
        ]

        # Get embeddings for all documents
        response = await client.aembedding(
            model="ollama/nomic-embed-text",
            input=documents
        )

        # Extract document embeddings
        document_embeddings = [item.embedding for item in response.data]

        # Search queries
        queries = [
            "Healthcare AI applications",
            "Financial market trends",
            "Medical image analysis"
        ]

        # Process each query
        for query in queries:
            print(f"\nQuery: {query}")

            # Get query embedding
            query_response = await client.aembedding(
                model="ollama/nomic-embed-text",
                input=query
            )
            query_embedding = query_response.data[0].embedding

            # Calculate similarity to each document
            similarities = []
            for i, doc_embedding in enumerate(document_embeddings):
                similarity = cosine_similarity(query_embedding, doc_embedding)
                similarities.append((i, similarity))

            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Print results
            print("Top matches:")
            for i, (doc_idx, similarity) in enumerate(similarities[:3]):
                print(f"  {i+1}. [{similarity:.5f}] {documents[doc_idx]}")


async def main():
    """Run all embedding examples"""
    await test_embeddings()
    await test_embedding_search()


if __name__ == "__main__":
    asyncio.run(main())
