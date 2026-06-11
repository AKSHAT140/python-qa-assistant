"""
ingest.py
---------
Step 2: Embed Q&A pairs with sentence-transformers → store in ChromaDB.
Output: chroma_db/  (vector index - commit this folder to GitHub!)

Run once locally after preprocess.py:
    python ingest.py
"""

import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import sys

PARQUET_PATH = "data/clean_qa.parquet"
CHROMA_PATH  = "chroma_db"
COLLECTION   = "python_qa"
BATCH_SIZE   = 128
MODEL_NAME   = "all-MiniLM-L6-v2"   # ~90MB, downloads once, very fast at inference

# ─── Guard: skip if already ingested ─────────────────────────────────────────
if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
    print(f"✅ ChromaDB already exists at '{CHROMA_PATH}'. Skipping ingestion.")
    print("   Delete the folder and re-run to re-index.")
    sys.exit(0)

# ─── Load data ────────────────────────────────────────────────────────────────
if not os.path.exists(PARQUET_PATH):
    print("❌ data/clean_qa.parquet not found. Run preprocess.py first.")
    sys.exit(1)

print(f"📂 Loading {PARQUET_PATH}...")
df = pd.read_parquet(PARQUET_PATH)
print(f"   ✅ {len(df):,} Q&A pairs loaded")

# ─── Build the document string that will be embedded ─────────────────────────
# Format: "Q: <title>\n<question_body>\nA: <answer_body>"
# We cap lengths so single docs don't blow up memory
df["document"] = df.apply(
    lambda r: (
        f"Q: {r['title']}\n"
        f"{str(r['question_body'])[:600]}\n\n"
        f"A: {str(r['answer_body'])[:900]}"
    ),
    axis=1,
)

# ─── Load embedding model ─────────────────────────────────────────────────────
print(f"\n🤖 Loading embedding model '{MODEL_NAME}'...")
print("   (Downloads ~90MB on first run, cached afterwards)")
model = SentenceTransformer(MODEL_NAME)
print("   ✅ Model ready")

# ─── Setup ChromaDB ───────────────────────────────────────────────────────────
print(f"\n Setting up ChromaDB at '{CHROMA_PATH}'...")
client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False)
)

# Fresh collection
try:
    client.delete_collection(COLLECTION)
except Exception:
    pass

collection = client.create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"},   # cosine similarity for semantic search
)

# ─── Embed & Index in batches ─────────────────────────────────────────────────
print(f"\n⚙️  Embedding and indexing {len(df):,} documents (batch={BATCH_SIZE})...")
print("   This takes ~5-15 minutes on CPU. Grab a coffee ☕")

total_indexed = 0
for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Indexing"):
    batch = df.iloc[i : i + BATCH_SIZE]

    docs   = batch["document"].tolist()
    ids    = [str(qid) for qid in batch["question_id"].tolist()]
    metas  = [
        {
            "title":          str(row["title"])[:200],
            "question_score": int(row["question_score"]),
            "answer_score":   int(row["answer_score"]),
        }
        for _, row in batch.iterrows()
    ]

    embeddings = model.encode(docs, show_progress_bar=False).tolist()

    collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=ids,
        metadatas=metas,
    )
    total_indexed += len(batch)

print(f"\n✅ Done! Indexed {collection.count():,} documents into ChromaDB")
print(f"   Folder: {os.path.abspath(CHROMA_PATH)}")
print("\n📌 Next step: commit the 'chroma_db/' folder to your GitHub repo")
print("   Then deploy to Render — no re-indexing needed!")
