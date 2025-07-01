
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from pinecone import Pinecone
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WSE Insight API", description="Wave Swell Energy Technical Assistant")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []

# Global variables for RAG components
query_engine = None
vector_store = None

def initialize_rag():
    """Initialize RAG components with Pinecone and LlamaIndex"""
    global query_engine, vector_store
    
    try:
        # Get environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "llamaindex")
        
        if not openai_api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            return False
            
        if not pinecone_api_key:
            logger.error("PINECONE_API_KEY not found in environment variables")
            return False
        
        # Configure LlamaIndex settings
        Settings.llm = OpenAI(
            model="gpt-3.5-turbo",
            api_key=openai_api_key,
            temperature=0.3
        )
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=openai_api_key,
            dimensions=1024
        )
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Connect to existing index
        pinecone_index = pc.Index(index_name)
        
        # Create vector store
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        
        # Create index from existing vector store
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        # Create query engine
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="tree_summarize"
        )
        
        logger.info(f"RAG system initialized successfully with index: {index_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    success = initialize_rag()
    if not success:
        logger.warning("RAG system initialization failed - chat functionality may be limited")

@app.get("/")
async def serve_frontend():
    """Serve the HTML frontend"""
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_initialized": query_engine is not None
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Handle chat requests with RAG"""
    try:
        if not query_engine:
            raise HTTPException(
                status_code=503,
                detail="RAG system not initialized. Please check your environment variables."
            )
        
        # Query the RAG system
        response = query_engine.query(request.message)
        
        # Extract sources from response
        sources = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes:
                if hasattr(node, 'metadata') and 'file_name' in node.metadata:
                    sources.append(node.metadata['file_name'])
                elif hasattr(node, 'metadata') and 'source' in node.metadata:
                    sources.append(node.metadata['source'])
        
        # Remove duplicates while preserving order
        sources = list(dict.fromkeys(sources))
        
        return ChatResponse(
            answer=str(response),
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat request: {str(e)}"
        )

# Mount static files for CSS and JS
app.mount("/static", StaticFiles(directory="."), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
