# RAG Query System
# This script lets you ask questions about your indexed documents

import os
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone

# Your API keys (replace with your actual keys)
OPENAI_API_KEY = "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"
PINECONE_API_KEY = "pcsk_69UEXW_KjZyGSc9VNoypPtA8E4QNRKwzBpmRf8PuNuQm7QTT82y6jZyS1TwTyLU7GpgYNi"

# Set environment variables
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Pinecone settings
PINECONE_INDEX_NAME = "llamaindex"

def setup_llama_index():
    """Configure LlamaIndex settings"""
    print("🔧 Setting up LlamaIndex...")
    
    # Configure LLM (language model)
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    # Configure embedding model - must match what you used for ingestion
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=1024)
    
    print("✅ LlamaIndex configured!")

def connect_to_index():
    """Connect to your existing Pinecone index"""
    print("🔧 Connecting to your document index...")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Connect to your existing index
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    
    # Create vector store
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Load the existing index
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    
    print("✅ Connected to index!")
    # Get vector count
    stats = pinecone_index.describe_index_stats()
    vector_count = stats.get('total_vector_count', 0)
    print(f"📊 Your index contains {vector_count} vectors")
    
    return index

def create_query_engine(index):
    """Create a query engine for asking questions"""
    print("🔧 Setting up query engine...")
    
    # Create a custom system prompt for professional responses
    system_prompt = """You are a technical assistant with access to company documents. 

    Your role is to provide precise, factual answers based ONLY on the retrieved document content.

    Guidelines:
    - Be concise and technical in your responses
    - Use **bold** for key technical terms, numbers, and important concepts
    - Structure your answers clearly with paragraphs for readability
    - If the documents don't contain sufficient information, state: "The answer is not available in the provided documents"
    - Stay professional and avoid speculation

    Format your responses to be helpful and authoritative while staying within the bounds of the retrieved information."""

# Create query engine with custom settings and prompt
    query_engine = index.as_query_engine(
        similarity_top_k=5,  # Number of similar chunks to retrieve
        response_mode="compact",  # How to combine information
        system_prompt=system_prompt
    )

    print("✅ Query engine ready!")
    return query_engine

def interactive_query_loop(query_engine):
    """Interactive loop for asking questions"""
    print("\n" + "="*60)
    print("🎉 RAG SYSTEM READY!")
    print("="*60)
    print("You can now ask questions about your documents!")
    print("Type 'quit' or 'exit' to stop.")
    print("="*60)
    
    while True:
        # Get user question
        print("\n📝 Your question:")
        question = input("> ").strip()
        
        # Check for exit commands
        if question.lower() in ['quit', 'exit', 'q', '']:
            print("\n👋 Goodbye!")
            break
        
        # Process the question
        print("\n🔍 Searching your documents...")
        try:
            response = query_engine.query(question)
            
            print("\n📋 Answer:")
            print("-" * 40)
            print(response.response)
            
            # Show sources if available (deduplicated by filename)
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print("\n📚 Sources:")
                # Get unique filenames to avoid duplicates
                unique_sources = set()
                for node in response.source_nodes:
                    filename = node.metadata.get('file_name', 'Unknown')
                    unique_sources.add(filename)
                
                # Display unique sources
                for i, filename in enumerate(sorted(unique_sources), 1):
                    print(f"   {i}. {filename}")
            
            print("-" * 40)
            
        except Exception as e:
            print(f"\n❌ Error processing question: {e}")
            print("Please try rephrasing your question.")

def main():
    """Main query system"""
    print("🚀 Starting RAG Query System...\n")
    
    # Setup
    setup_llama_index()
    
    # Connect to your indexed documents
    try:
        index = connect_to_index()
    except Exception as e:
        print(f"❌ Error connecting to index: {e}")
        print("Make sure you've run the ingestion script first!")
        return
    
    # Create query engine
    query_engine = create_query_engine(index)
    
    # Start interactive session
    interactive_query_loop(query_engine)

if __name__ == "__main__":
    main()