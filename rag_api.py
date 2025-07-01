# Secure RAG API Backend
# A production-ready FastAPI server for your RAG system

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# FastAPI and security imports
from fastapi import FastAPI, HTTPException, Depends, Security, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import uvicorn

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Security configuration
security = HTTPBearer()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") 
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "llamaindex")
API_KEYS = os.getenv("VALID_API_KEYS", "wse-demo-key-2025").split(",")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "https://*.replit.app,http://localhost:*").split(",")

# Rate limiting storage
request_counts = {}
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 3600

# Global variables
query_engine = None
startup_time = None

class ChatRequest(BaseModel):
    """Request model for chat queries"""
    message: str
    max_tokens: Optional[int] = 1000
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        if len(v) > 2000:
            raise ValueError('Message too long (max 2000 characters)')
        if any(char in v for char in ['<script>', 'javascript:', 'vbscript:']):
            raise ValueError('Invalid characters in message')
        return v.strip()

class ChatResponse(BaseModel):
    """Response model for chat queries"""
    answer: str
    sources: list
    query_time: float
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    uptime_seconds: float
    system_info: Dict[str, Any]

def check_rate_limit(client_ip: str) -> bool:
    """Simple rate limiting by IP address"""
    current_time = time.time()
    
    # Clean old entries
    for ip in list(request_counts.keys()):
        request_counts[ip] = [req_time for req_time in request_counts[ip] 
                            if current_time - req_time < RATE_LIMIT_WINDOW]
        if not request_counts[ip]:
            del request_counts[ip]
    
    # Check current IP
    if client_ip not in request_counts:
        request_counts[client_ip] = []
    
    if len(request_counts[client_ip]) >= RATE_LIMIT_REQUESTS:
        return False
    
    request_counts[client_ip].append(current_time)
    return True

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key from Authorization header"""
    if credentials.credentials not in API_KEYS:
        logger.warning(f"Invalid API key attempted: {credentials.credentials[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

async def setup_rag_system():
    """Initialize the RAG system"""
    global query_engine
    
    try:
        logger.info("Setting up RAG system...")
        
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
        
        Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.3)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=1024)
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        
        stats = pinecone_index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0)
        logger.info(f"Connected to Pinecone index with {vector_count} vectors")
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        
        system_prompt = """You are a technical assistant for Wave Swell Energy (WSE), a renewable energy company specializing in wave energy conversion systems.

Your role is to provide precise, factual answers based ONLY on the retrieved document content.

Guidelines:
- Be concise and technical in your responses
- Use **bold** for key technical terms, numbers, and important concepts
- Structure your answers clearly with paragraphs for readability
- If the documents don't contain sufficient information, state: "The answer is not available in the provided documents"
- Stay professional and avoid speculation

Format your responses to be helpful and authoritative while staying within the bounds of the retrieved information."""

        query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
            system_prompt=system_prompt
        )
        
        logger.info("RAG system initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to setup RAG system: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    global startup_time
    startup_time = time.time()
    
    logger.info("Starting WSE RAG API Server...")
    await setup_rag_system()
    logger.info("Server startup complete")
    
    yield
    
    logger.info("Shutting down WSE RAG API Server...")

# Create FastAPI app
app = FastAPI(
    title="WSE Technical Assistant API",
    description="Secure API for Wave Swell Energy technical document queries",
    version="1.0.0",
    lifespan=lifespan
)

# Security Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "*.replit.app", "*.replit.dev"]
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting by IP address"""
    client_ip = request.client.host
    
    if not check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again later."}
        )
    
    return await call_next(request)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    current_time = time.time()
    uptime = current_time - startup_time if startup_time else 0
    
    try:
        test_available = query_engine is not None
        
        system_info = {
            "rag_system": "operational" if test_available else "unavailable",
            "openai_configured": bool(OPENAI_API_KEY),
            "pinecone_configured": bool(PINECONE_API_KEY),
            "api_version": "1.0.0"
        }
        
        status_value = "healthy" if test_available else "degraded"
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        system_info = {"error": str(e)}
        status_value = "unhealthy"
    
    return HealthResponse(
        status=status_value,
        timestamp=datetime.now().isoformat(),
        uptime_seconds=uptime,
        system_info=system_info
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_query(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key),
    client_request: Request = None
):
    """Main chat endpoint for document queries"""
    start_time = time.time()
    client_ip = client_request.client.host if client_request else "unknown"
    
    try:
        logger.info(f"Query from IP {client_ip}: {request.message[:100]}...")
        
        if not query_engine:
            raise HTTPException(
                status_code=503,
                detail="RAG system not available"
            )
        
        response = query_engine.query(request.message)
        
        unique_sources = set()
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes:
                filename = node.metadata.get('file_name', 'Unknown')
                unique_sources.add(filename)
        
        sources_list = sorted(list(unique_sources))
        query_time = time.time() - start_time
        
        logger.info(f"Query completed in {query_time:.2f}s, {len(sources_list)} sources")
        
        return ChatResponse(
            answer=str(response.response),
            sources=sources_list,
            query_time=query_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "WSE Technical Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "docs": "/docs"
        }
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

if __name__ == "__main__":
    uvicorn.run(
        "rag_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )