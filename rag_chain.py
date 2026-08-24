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
PRIMARY_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
TOP_K         = 5    # How many SO Q&As to retrieve as context

# Candidate fallback models if configured model is unavailable or deprecated
CANDIDATE_MODELS = [
    PRIMARY_GROQ_MODEL,
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]
FALLBACK_MODELS = list(dict.fromkeys([m for m in CANDIDATE_MODELS if m]))

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

print(f"⚡ Initializing Groq client...")
_groq = Groq(api_key=GROQ_API_KEY)

# Dynamically discover active models if API allows
_active_models = []
try:
    models_resp = _groq.models.list()
    _active_models = [m.id for m in models_resp.data if hasattr(m, 'id')]
    print(f"   Found {len(_active_models)} available Groq models: {_active_models}")
except Exception as e:
    print(f"   ⚠️ Could not fetch Groq model list dynamically: {e}")

# Determine starting model
if PRIMARY_GROQ_MODEL in _active_models:
    GROQ_MODEL = PRIMARY_GROQ_MODEL
elif _active_models:
    preferred = [m for m in _active_models if any(k in m.lower() for k in ["llama", "mixtral", "gemma"])]
    GROQ_MODEL = preferred[0] if preferred else _active_models[0]
    print(f"   ⚠️ Model '{PRIMARY_GROQ_MODEL}' not in active list. Using '{GROQ_MODEL}'.")
else:
    GROQ_MODEL = PRIMARY_GROQ_MODEL

print(f"   ✅ Primary model set to: {GROQ_MODEL}")

# ─── System prompt ───────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are an expert Python programming assistant powered by real Stack Overflow data.

Rules:
1. Answer ONLY based on the Stack Overflow context provided below.
2. If the context does not fully cover the question, say so briefly, then answer from your knowledge.
3. Always include working code examples in Python using markdown code blocks (```python ... ```).
4. Be concise but complete — avoid unnecessary padding.
5. If multiple approaches exist, list them with trade-offs.
"""

def _call_groq_with_fallback(messages: list) -> tuple[str, str]:
    """
    Calls Groq chat completion API with automatic model fallback
    if the chosen model is not found or encounters a 404 error.

    Returns: (answer_text, model_used)
    """
    global GROQ_MODEL

    models_to_try = list(dict.fromkeys([GROQ_MODEL] + _active_models + FALLBACK_MODELS))

    last_exception = None
    for model_name in models_to_try:
        try:
            response = _groq.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content
            GROQ_MODEL = model_name
            return answer, model_name
        except Exception as e:
            err_str = str(e)
            if "404" in err_str or "model_not_found" in err_str or "does not exist" in err_str:
                print(f"⚠️ Model '{model_name}' not available on Groq (404/model_not_found). Trying fallback...")
                last_exception = e
                continue
            else:
                raise e

    raise RuntimeError(
        f"All attempted Groq models ({models_to_try}) failed. Last error: {last_exception}"
    )

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

    # 4️⃣ Build prompt and call Groq with fallback
    user_message = (
        f"Python Question: {question}\n\n"
        f"Stack Overflow Context:\n{context_str}\n\n"
        f"Please answer the question based on the context above."
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    answer, model_used = _call_groq_with_fallback(messages)

    return {
        "question":       question,
        "answer":         answer,
        "sources":        source_titles,
        "model":          model_used,
        "docs_retrieved": TOP_K,
    }

