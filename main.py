"""
main.py
-------
FastAPI application — the public-facing API.

Endpoints:
    GET  /           → API info
    GET  /health     → health check (required by assessment)
    POST /ask        → ask a Python question (required by assessment)

Run locally:
    uvicorn main:app --reload --port 8000
"""

import time
import os
import zipfile
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import os
import zipfile

# ─── Ensure Database is Unzipped BEFORE loading RAG pipeline ──────────────────
ZIP_PATH = "chroma_db.zip"
EXTRACT_PATH = "chroma_db"
DB_FOLDER = "chroma_db"

if os.path.exists(ZIP_PATH) and not os.path.exists(DB_FOLDER):
    print(f"📦 Unzipping {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
    print("✅ Unzipping complete.")

import rag_chain   # Importing this triggers model/DB load at startup

# ─── App state ────────────────────────────────────────────────────────────────
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("🚀 Starting Python Q&A Assistant...")
    _state["start_time"] = time.time()
    _state["request_count"] = 0
    yield
    print("👋 Shutting down.")


# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Python Q&A Assistant",
    description=(
        "AI-powered Python programming Q&A backed by real Stack Overflow data. "
        "Uses a RAG pipeline: sentence-transformers + ChromaDB + Groq LLaMA-3.3-70b."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ──────────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        example="How do I reverse a list in Python?",
    )


class AnswerResponse(BaseModel):
    question:       str
    answer:         str
    sources:        List[str] = Field(description="Stack Overflow question titles used as context")
    model:          str       = Field(description="LLM model used to generate the answer")
    docs_retrieved: int       = Field(description="Number of SO documents retrieved as context")


class HealthResponse(BaseModel):
    status:          str
    uptime_seconds:  float
    requests_served: int
    index_size:      int


# ─── Middleware: count requests ───────────────────────────────────────────────
@app.middleware("http")
async def count_requests(request: Request, call_next):
    _state["request_count"] = _state.get("request_count", 0) + 1
    return await call_next(request)


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    """API overview."""
    return {
        "name":        "Python Q&A Assistant",
        "description": "Ask any Python programming question and get answers grounded in Stack Overflow data.",
        "version":     "1.0.0",
        "endpoints": {
            "POST /ask":    "Ask a Python question",
            "GET  /health": "Health check",
            "GET  /docs":   "Interactive Swagger docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """
    Health check endpoint.
    Returns API status, uptime, request count, and index size.
    """
    uptime = time.time() - _state.get("start_time", time.time())
    try:
        index_size = rag_chain._collection.count()
    except Exception:
        index_size = -1

    return HealthResponse(
        status="ok",
        uptime_seconds=round(uptime, 2),
        requests_served=_state.get("request_count", 0),
        index_size=index_size,
    )


@app.post("/ask", response_model=AnswerResponse, tags=["Q&A"])
def ask(request: QuestionRequest):
    """
    Ask a Python programming question.

    The system will:
    1. Embed your question using sentence-transformers
    2. Retrieve the 5 most relevant Stack Overflow Q&As
    3. Feed them as context to Groq LLaMA-3.3-70b
    4. Return a grounded, accurate answer

    **Example:**
    ```json
    { "question": "How do I read a CSV file with pandas?" }
    ```
    """
    q = request.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = rag_chain.get_answer(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")

    return AnswerResponse(**result)
