# RAG Document Ingestion Script
# This script loads documents, processes them, and stores them in Pinecone

import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
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
PINECONE_INDEX_NAME = "llamaindex"  # The index you created

def setup_llama_index():
    """Configure LlamaIndex settings"""
    print("🔧 Setting up LlamaIndex...")
    
    # Configure LLM (language model)
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    
    # Configure embedding model - Use text-embedding-3-small with dimensions=1024 to match your Pinecone index
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=1024)
    
    print("✅ LlamaIndex configured!")

def setup_pinecone(clear_index=False):
    """Setup Pinecone vector store"""
    print("🔧 Setting up Pinecone connection...")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Connect to your index
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    
    # Option to clear existing data
    if clear_index:
        print("🗑️  Clearing existing data from index...")
        pinecone_index.delete(delete_all=True)
        print("✅ Index cleared!")
    
    # Create vector store
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    print("✅ Pinecone connected!")
    return vector_store

def load_documents(documents_path="documents"):
    """Load documents from the specified folder"""
    print(f"📄 Loading documents from '{documents_path}' folder...")
    
    # Check if folder exists
    if not os.path.exists(documents_path):
        print(f"❌ Error: '{documents_path}' folder not found!")
        print(f"Please create a '{documents_path}' folder and add your documents to it.")
        return None
    
    # Load documents
    try:
        reader = SimpleDirectoryReader(documents_path)
        documents = reader.load_data()
        
        print(f"✅ Loaded {len(documents)} documents:")
        for i, doc in enumerate(documents, 1):
            # Get filename from metadata if available
            filename = doc.metadata.get('file_name', f'Document {i}')
            print(f"   {i}. {filename}")
        
        return documents
    
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        return None

def create_index(documents, vector_store):
    """Create and populate the vector index"""
    print("🔄 Processing documents and creating index...")
    print("   (This may take a few minutes depending on document size)")
    
    try:
        # Create storage context with Pinecone
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create the index
        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context,
            show_progress=True
        )
        
        print("✅ Index created successfully!")
        return index
    
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        return None

def main():
    """Main ingestion process"""
    print("🚀 Starting Document Ingestion Process...\n")
    
    # Setup (with option to clear existing data)
    setup_llama_index()
    
    # Ask user if they want to clear existing data
    clear_data = input("⚠️  Clear existing data before ingesting? (y/N): ").lower().strip()
    should_clear = clear_data in ['y', 'yes']
    
    vector_store = setup_pinecone(clear_index=should_clear)
    
    # Load documents
    documents = load_documents()
    if not documents:
        return
    
    # Create index
    index = create_index(documents, vector_store)
    if not index:
        return
    
    # Success!
    print("\n" + "="*50)
    print("🎉 INGESTION COMPLETE!")
    print("="*50)
    print(f"✅ Successfully processed {len(documents)} documents")
    print(f"✅ Documents are now searchable in Pinecone index: {PINECONE_INDEX_NAME}")
    print("\nYou can now run the query script to ask questions about your documents!")

if __name__ == "__main__":
    main()