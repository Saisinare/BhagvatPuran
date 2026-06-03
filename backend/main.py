"""
RAGita Backend API - FastAPI server for Bhagavad Gita RAG system
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import sys
from pathlib import Path
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from RAG.pipeline.orchestrator import answer_query, detect_query_type_and_persona


# ============= Lifespan Event Handler =============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan event handler (replaces deprecated on_event)"""
    # Startup
    print("🚀 RAGita API starting up...")
    print("📚 Loading Bhagavad Gita knowledge base...")
    # Warm up the retriever
    try:
        from RAG.pipeline.retriever import Retriever
        retriever = Retriever()
        print(f"✅ Indexed {len(retriever.metas)} verses")
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize retriever: {e}")
    print("🎉 RAGita API ready!")
    
    yield
    
    # Shutdown
    print("👋 RAGita API shutting down...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="RAGita API",
    description="Bhagavad Gita Question Answering API with RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= Request/Response Models =============

class QueryRequest(BaseModel):
    query: str = Field(..., description="User's question about Bhagavad Gita", min_length=3)
    persona: Optional[str] = Field("versatile", description="Response style: versatile/scholarly/devotional/practical/philosophical/counselor/teacher")
    top_k: Optional[int] = Field(15, description="Number of passages to retrieve", ge=5, le=30)
    rerank_k: Optional[int] = Field(8, description="Number of passages to rerank", ge=3, le=15)

class Citation(BaseModel):
    verse: str
    passage_preview: str

class VerificationDetail(BaseModel):
    claim: str
    support_score: float
    supported: bool
    best_passage_verse: Optional[str]
    best_passage_preview: Optional[str]

class QueryResponse(BaseModel):
    query: str
    answer: str
    persona: str
    query_type: str
    response_time: float
    citations: List[Citation]
    verification_summary: Dict[str, Any]
    retrieved_docs: List[str]
    confidence_score: float

class HealthResponse(BaseModel):
    status: str
    message: str
    version: str

class StatsResponse(BaseModel):
    total_verses: int
    indexed_verses: int
    available_personas: List[str]
    system_status: str

# ============= API Endpoints =============

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - API health check"""
    return {
        "status": "healthy",
        "message": "RAGita API is running. Visit /docs for API documentation.",
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "All systems operational",
        "version": "1.0.0"
    }

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a user query and return an answer with citations
    
    - **query**: The question to ask about the Bhagavad Gita
    - **persona**: Response style (default: versatile)
    - **top_k**: Number of passages to retrieve (default: 15)
    - **rerank_k**: Number of passages to rerank (default: 8)
    """
    try:
        start_time = time.time()
        
        # Process query through RAG pipeline
        result = answer_query(
            query=request.query,
            persona=request.persona,
            top_k=request.top_k,
            rerank_k=request.rerank_k
        )
        
        # Extract verification data
        verification = result.get("verification", [])
        total_claims = len(verification)
        supported_claims = sum(1 for v in verification if v.get("supported", False))
        
        # Calculate confidence score
        confidence_score = supported_claims / total_claims if total_claims > 0 else 1.0
        
        # Format citations
        citations = [
            Citation(
                verse=c.get("verse", ""),
                passage_preview=c.get("passage_preview", "")
            )
            for c in result.get("citations", [])
        ]
        
        # Format response
        response = QueryResponse(
            query=request.query,
            answer=result.get("answer_raw", ""),
            persona=result.get("persona", request.persona),
            query_type=result.get("query_type", "general"),
            response_time=time.time() - start_time,
            citations=citations,
            verification_summary={
                "total_claims": total_claims,
                "supported_claims": supported_claims,
                "unsupported_claims": result.get("unsupported_claims", 0),
                "grounding_accuracy": confidence_score
            },
            retrieved_docs=[
                f"{c.get('chapter')}:{c.get('verse')}"
                for c in result.get("candidates", [])
            ],
            confidence_score=confidence_score
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/detect-query-type")
async def detect_type(query: str):
    """
    Detect the type of query and suggest appropriate persona
    
    - **query**: The question to analyze
    """
    try:
        persona, query_type = detect_query_type_and_persona(query)
        return {
            "query": query,
            "detected_persona": persona,
            "query_type": query_type,
            "description": get_persona_description(persona)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting query type: {str(e)}")

@app.get("/personas")
async def get_personas():
    """Get list of available response personas with descriptions"""
    return {
        "personas": [
            {
                "id": "versatile",
                "name": "Versatile",
                "description": "Balanced approach suitable for general queries"
            },
            {
                "id": "scholarly",
                "name": "Scholarly",
                "description": "Academic and analytical responses with deep philosophical insight"
            },
            {
                "id": "devotional",
                "name": "Devotional",
                "description": "Spiritual and devotional tone for worship and bhakti"
            },
            {
                "id": "practical",
                "name": "Practical",
                "description": "Action-oriented guidance for daily life application"
            },
            {
                "id": "philosophical",
                "name": "Philosophical",
                "description": "Deep philosophical exploration of concepts"
            },
            {
                "id": "counselor",
                "name": "Counselor",
                "description": "Compassionate guidance for personal challenges"
            },
            {
                "id": "teacher",
                "name": "Teacher",
                "description": "Educational explanations for learning and understanding"
            }
        ]
    }

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    try:
        from RAG.pipeline.retriever import Retriever
        retriever = Retriever()
        
        return {
            "total_verses": 701,
            "indexed_verses": len(retriever.metas),
            "available_personas": ["versatile", "scholarly", "devotional", "practical", "philosophical", "counselor", "teacher"],
            "system_status": "operational"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

# ============= Helper Functions =============

def get_persona_description(persona: str) -> str:
    """Get description for a persona"""
    descriptions = {
        "versatile": "Balanced approach suitable for general queries",
        "scholarly": "Academic and analytical responses with deep philosophical insight",
        "devotional": "Spiritual and devotional tone for worship and bhakti",
        "practical": "Action-oriented guidance for daily life application",
        "philosophical": "Deep philosophical exploration of concepts",
        "counselor": "Compassionate guidance for personal challenges",
        "teacher": "Educational explanations for learning and understanding"
    }
    return descriptions.get(persona, "General response style")


# ============= Main Entry Point =============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

