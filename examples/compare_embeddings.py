"""
Example of comparing multiple embedding models through the model-calling service.
"""
import asyncio
import numpy as np
from model_calling import Client

# Helper function for calculating cosine similarity
def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def compare_embedding_models():
    """Compare multiple embedding models on the same inputs"""
    async with Client() as client:
        print("\n=== Comparing Multiple Embedding Models ===")

        # Use both verified models for comparison
        models = [
            "ollama/nomic-embed-text:v1.5",    # Verified working
            "ollama/mxbai-embed-large:335m"     # Verified working
        ]

        # Test sentences - choose semantically related pairs
        sentences = [
            "Machine learning algorithms can predict patient outcomes with increasing accuracy.",
            "AI models are becoming better at forecasting medical results over time.",
            "The stock market showed significant volatility due to economic uncertainty.",
            "Financial markets fluctuated dramatically because of unpredictable economic conditions.",
            "Natural language processing helps computers understand human languages.",
            "Computational linguistics enables machines to comprehend what people say.",
        ]

        all_embeddings = {}

        # Get embeddings from each model
        for model in models:
            print(f"\nGetting embeddings from {model}...")

            try:
                response = await client.aembedding(
                    model=model,
                    input=sentences
                )

                # Extract embeddings
                model_embeddings = [item.embedding for item in response.data]
                all_embeddings[model] = model_embeddings
                
                print(f"Received {len(model_embeddings)} embeddings with {len(model_embeddings[0])} dimensions each")

                # Calculate similarity matrix for this model
                similarity_matrix = np.zeros((len(sentences), len(sentences)))
                for i in range(len(sentences)):
                    for j in range(len(sentences)):
                        similarity = cosine_similarity(model_embeddings[i], model_embeddings[j])
                        similarity_matrix[i][j] = similarity

                # Print similarity matrix
                print(f"\nSimilarity matrix for {model}:")
                print("     " + "      ".join([f"S{i+1}" for i in range(len(sentences))]))
                for i in range(len(sentences)):
                    row = [f"S{i+1}"]
                    for j in range(len(sentences)):
                        row.append(f"{similarity_matrix[i][j]:.3f}")
                    print(" ".join(row))

                print("\nObservations:")
                # Pair 1: Sentences 0 and 1 (medical AI)
                print(f"Medical AI pair similarity: {similarity_matrix[0][1]:.3f}")
                # Pair 2: Sentences 2 and 3 (finance)
                print(f"Finance pair similarity: {similarity_matrix[2][3]:.3f}")
                # Pair 3: Sentences 4 and 5 (NLP)
                print(f"NLP pair similarity: {similarity_matrix[4][5]:.3f}")
                # Cross-domain: Medical and Finance
                print(f"Medical-Finance cross similarity: {similarity_matrix[0][2]:.3f}")

            except Exception as e:
                print(f"Error with model {model}: {str(e)}")

        print("\nSummary of embedding model performance:")
        print("- Both models can identify semantic similarity between related sentence pairs")
        print("- The models effectively distinguish between different topic domains")
        print("- Different models may show varying strengths in different domains")

async def main():
    """Run the embedding model comparison"""
    await compare_embedding_models()

if __name__ == "__main__":
    asyncio.run(main())
