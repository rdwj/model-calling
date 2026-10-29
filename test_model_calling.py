#!/usr/bin/env python3
"""
Test script for the model-calling package
"""

from model_calling.client import SyncModelCallingClient

def test_basic_chat():
    """Test basic chat completion"""
    client = SyncModelCallingClient()
    
    try:
        print("Testing basic chat completion...")
        response = client.chat_completion(
            model="ollama/llama3:8b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is machine learning? Give a brief explanation."}
            ],
            temperature=0.7
        )
        
        print("Response:")
        print(response["choices"][0]["message"]["content"])
        print("-" * 50)
        
    except Exception as e:
        print(f"Error in basic chat: {e}")
    finally:
        client.close()

def test_list_models():
    """Test listing available models"""
    client = SyncModelCallingClient()
    
    try:
        print("Testing model listing...")
        models = client.list_models()
        print("Available models:")
        for model in models.get("data", []):
            print(f"  - {model.get('id', 'Unknown')}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error listing models: {e}")
    finally:
        client.close()

def test_embeddings():
    """Test embeddings functionality"""
    client = SyncModelCallingClient()
    
    try:
        print("Testing embeddings...")
        response = client.embeddings(
            model="ollama/nomic-embed-text",
            input="This is a test sentence for embeddings."
        )
        
        embeddings = response["data"][0]["embedding"]
        print(f"Generated embedding with {len(embeddings)} dimensions")
        print(f"First 5 values: {embeddings[:5]}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error with embeddings: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    print("Model Calling Package Test")
    print("=" * 50)
    
    test_list_models()
    test_basic_chat()
    test_embeddings()
    
    print("Test completed!")