# Python Q&A Assistant

An AI-powered Python programming question-answering system backed by real **Stack Overflow data**.

Built with **FastAPI** + **LangChain RAG** + **ChromaDB** + **Groq LLaMA-3.3-70b**.

---

## 🏗️ Architecture

```
User Question
    │
    ▼
[sentence-transformers]   ← embeds question into a vector
    │
    ▼
[ChromaDB Vector Index]   ← retrieves top-5 similar Stack Overflow Q&As
    │
    ▼
[Groq LLaMA-3.3-70b]     ← generates grounded answer from SO context
    │
    ▼
FastAPI  POST /ask        ← returns answer + sources as JSON
```

---

## 📁 Project Structure

```
Assessment/
├── preprocess.py      # Step 1: clean the raw CSV dataset → data/clean_qa.parquet
├── ingest.py          # Step 2: embed + index into ChromaDB → chroma_db/
├── rag_chain.py       # Core RAG pipeline (embedder + retriever + LLM)
├── main.py            # FastAPI app (POST /ask, GET /health)
├── test_queries.py    # Test suite (10 queries)
├── requirements.txt
├── .env.example       # Copy to .env and fill in your key
├── data/
│   └── clean_qa.parquet   # Pre-processed Q&A pairs (committed)
└── chroma_db/             # Vector index (committed — no re-indexing on deploy)
```

---

## ⚙️ Local Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/AKSHAT140/python-qa-assistant.git
cd python-qa-assistant

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

Get a free Groq key at: https://console.groq.com

### 4. (First time only) Preprocess + index the dataset

> **Skip this if `data/clean_qa.parquet` and `chroma_db/` already exist in the repo.**

```bash
# Requires the raw dataset in dataset/archive/
python preprocess.py    # ~5 min — cleans 15K Python Q&A pairs
python ingest.py        # ~10 min — embeds and indexes into ChromaDB
```

### 5. Run the API

```bash
uvicorn main:app --reload --port 8000
```

API is live at: http://localhost:8000  
Swagger docs: http://localhost:8000/docs

---

## 📡 API Endpoints

### `GET /health`

```json
{
  "status": "ok",
  "uptime_seconds": 42.3,
  "requests_served": 7,
  "index_size": 14823
}
```

### `POST /ask`

**Request:**
```json
{
  "question": "How do I sort a dictionary by value in Python?"
}
```

**Response:**
```json
{
  "question": "How do I sort a dictionary by value in Python?",
  "answer": "You can sort a dictionary by value using `sorted()` with a lambda...\n```python\nsorted_dict = dict(sorted(my_dict.items(), key=lambda x: x[1]))\n```",
  "sources": [
    "Sort a Python dictionary by value",
    "How do I sort a list of dictionaries by a value?"
  ],
  "model": "llama-3.3-70b-versatile",
  "docs_retrieved": 5
}
```

---

## 🧪 Running Tests

```bash
# In a second terminal (API must be running)
python test_queries.py
```

Results saved to `test_results.json`.

---

## 🚀 Deployment (Render)

1. Push your repo to GitHub (make sure `data/` and `chroma_db/` are included)
2. Create a new **Web Service** on [render.com](https://render.com)
3. Connect your GitHub repo
4. Set these values:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variable: `GROQ_API_KEY` = your key
6. Deploy!

**Live URL:** `https://your-app-name.onrender.com`

---

## 🛠️ Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | — | Groq API key |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Groq model name |
| `CHROMA_PATH` | No | `chroma_db` | Path to ChromaDB folder |

---

## 📊 Dataset

**Stack Overflow — Python Questions & Answers**  
Source: https://www.kaggle.com/datasets/stackoverflow/pythonquestions

- `Questions.csv` — 607K+ Python questions (filtered from ~1.5M total)
- `Answers.csv` — Corresponding answers with scores
- `Tags.csv` — Tags per question

We use the top **15,000 highest-scored** Python Q&As as our knowledge base.

## ?? Live Deployment

The application is actively deployed on Hugging Face Spaces (fulfilling the BONUS requirement).

**Deployed App URL:** [https://huggingface.co/spaces/akshat1409/python-qa-assistant](https://huggingface.co/spaces/akshat1409/python-qa-assistant)

*(Note: Both the FastAPI backend and Streamlit UI run concurrently on the space.)*

