"""
rag_chain.py
------------
Core RAG (Retrieval-Augmented Generation) pipeline.
Loaded once at startup by main.py — all objects are module-level singletons.

Flow:
    User question
        → embed with sentence-transformers
        → query ChromaDB for top-5 similar Q&As
        → build prompt with those Q&As as context
        → call Groq LLaMA-3.3-70b
        → return grounded answer
"""

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ─── Config ──────────────────────────────────────────────────────────────────
CHROMA_PATH   = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION    = "python_qa"
MODEL_NAME    = "all-MiniLM-L6-v2"
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
TOP_K         = 5    # How many SO Q&As to retrieve as context

if not GROQ_API_KEY:
    raise EnvironmentError("❌ GROQ_API_KEY not set. Add it to your .env file.")

# ─── Load once at module import (not per-request!) ───────────────────────────
print(f"🤖 Loading embedding model '{MODEL_NAME}'...")
_embedder = SentenceTransformer(MODEL_NAME)

print(f"Connecting to ChromaDB at '{CHROMA_PATH}'...")
_chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False)
)
_collection    = _chroma_client.get_collection(COLLECTION)
print(f"   ✅ {_collection.count():,} documents in index")

print(f"⚡ Initializing Groq client (model: {GROQ_MODEL})...")
_groq = Groq(api_key=GROQ_API_KEY)

# ─── System prompt ───────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are an expert Python programming assistant powered by real Stack Overflow data.

Rules:
1. Answer ONLY based on the Stack Overflow context provided below.
2. If the context does not fully cover the question, say so briefly, then answer from your knowledge.
3. Always include working code examples in Python using markdown code blocks (```python ... ```).
4. Be concise but complete — avoid unnecessary padding.
5. If multiple approaches exist, list them with trade-offs.
"""

# ─── Main function called by FastAPI ─────────────────────────────────────────
def get_answer(question: str) -> dict:
    """
    Parameters
    ----------
    question : str  — The user's Python question

    Returns
    -------
    dict with keys: question, answer, sources, model, docs_retrieved
    """
    # 1️⃣ Embed the question
    query_vec = _embedder.encode(question).tolist()

    # 2️⃣ Retrieve top-K similar Q&As from vector DB
    results = _collection.query(
        query_embeddings=[query_vec],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    # 3️⃣ Build context block from retrieved docs
    context_parts = []
    source_titles = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        title = meta.get("title", "Unknown")
        score = meta.get("question_score", 0)
        context_parts.append(
            f"--- Stack Overflow Result {i} ---\n"
            f"Title: {title} (votes: {score})\n"
            f"Relevance: {(1 - dist) * 100:.1f}%\n\n"
            f"{doc}"
        )
        source_titles.append(title)

    context_str = "\n\n".join(context_parts)

    # 4️⃣ Build prompt and call Groq
    user_message = (
        f"Python Question: {question}\n\n"
        f"Stack Overflow Context:\n{context_str}\n\n"
        f"Please answer the question based on the context above."
    )

    response = _groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.1,    # low temp = more factual, deterministic
        max_tokens=1024,
    )

    answer = response.choices[0].message.content

    return {
        "question":       question,
        "answer":         answer,
        "sources":        source_titles,
        "model":          GROQ_MODEL,
        "docs_retrieved": TOP_K,
    }
