# RAG Setup Test Script
# This script tests your OpenAI and Pinecone connections

import os
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Set your API keys here (replace with your actual keys)
OPENAI_API_KEY = "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"
PINECONE_API_KEY = "pcsk_69UEXW_KjZyGSc9VNoypPtA8E4QNRKwzBpmRf8PuNuQm7QTT82y6jZyS1TwTyLU7GpgYNi"

# Set environment variables
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

def test_openai_connection():
    """Test if OpenAI API is working"""
    print("Testing OpenAI connection...")
    try:
        # Initialize OpenAI LLM
        llm = OpenAI(model="gpt-3.5-turbo")
        
        # Test a simple completion
        response = llm.complete("Hello, this is a test. Please respond with 'Connection successful!'")
        print(f"✅ OpenAI Response: {response}")
        return True
    except Exception as e:
        print(f"❌ OpenAI Error: {e}")
        return False

def test_embeddings():
    """Test if OpenAI embeddings are working"""
    print("\nTesting OpenAI embeddings...")
    try:
        # Initialize embedding model
        embed_model = OpenAIEmbedding()
        
        # Test embedding creation
        embedding = embed_model.get_text_embedding("This is a test sentence for embedding.")
        print(f"✅ Embedding created successfully! Dimension: {len(embedding)}")
        return True
    except Exception as e:
        print(f"❌ Embedding Error: {e}")
        return False

def test_pinecone_connection():
    """Test if Pinecone API is working"""
    print("\nTesting Pinecone connection...")
    try:
        from pinecone import Pinecone
        
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # List indexes (this tests the connection)
        indexes = pc.list_indexes()
        print(f"✅ Pinecone connected! Available indexes: {[idx.name for idx in indexes]}")
        return True
    except Exception as e:
        print(f"❌ Pinecone Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting RAG Setup Tests...\n")
    
    # Run tests
    openai_ok = test_openai_connection()
    embeddings_ok = test_embeddings()
    pinecone_ok = test_pinecone_connection()
    
    # Summary
    print("\n" + "="*50)
    print("SETUP TEST SUMMARY:")
    print("="*50)
    print(f"OpenAI API: {'✅ Working' if openai_ok else '❌ Failed'}")
    print(f"Embeddings: {'✅ Working' if embeddings_ok else '❌ Failed'}")
    print(f"Pinecone: {'✅ Working' if pinecone_ok else '❌ Failed'}")
    
    if all([openai_ok, embeddings_ok, pinecone_ok]):
        print("\n🎉 All systems ready! You can build your RAG system now.")
    else:
        print("\n⚠️  Some connections failed. Check your API keys and try again.")

if __name__ == "__main__":
    main()