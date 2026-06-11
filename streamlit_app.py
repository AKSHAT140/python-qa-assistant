"""
streamlit_app.py
----------------
Beautiful Streamlit GUI for the Python Q&A Assistant.
Talks to the FastAPI backend running at localhost:8000.

Run with:
    streamlit run streamlit_app.py
(Make sure the FastAPI server is also running: uvicorn main:app --port 8000)
"""

import streamlit as st
import requests
import time
from datetime import datetime

# ─── Page config (MUST be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Python Q&A Assistant",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Main background ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1528 50%, #0a1020 100%);
    min-height: 100vh;
}

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #1a2744 0%, #0f1e3d 60%, #162040 100%);
    border: 1px solid rgba(99, 179, 237, 0.15);
    border-radius: 20px;
    padding: 36px 40px 28px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,179,237,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #63b3ed, #90cdf4, #4299e1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0;
    line-height: 1.2;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 400;
    margin: 0;
}

/* ── Question input card ── */
.question-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99, 179, 237, 0.2);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}

/* ── Answer card ── */
.answer-card {
    background: linear-gradient(135deg, rgba(26,39,68,0.8), rgba(15,30,61,0.9));
    border: 1px solid rgba(99, 179, 237, 0.25);
    border-left: 4px solid #4299e1;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 16px 0;
    backdrop-filter: blur(10px);
    animation: fadeIn 0.4s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.answer-text {
    color: #e2e8f0;
    font-size: 0.97rem;
    line-height: 1.75;
}

/* ── History item ── */
.history-item {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: all 0.2s;
}
.history-item:hover {
    border-color: rgba(99,179,237,0.3);
    background: rgba(99,179,237,0.05);
}
.history-q {
    color: #90cdf4;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 0 0 4px 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.history-time {
    color: #4a5568;
    font-size: 0.75rem;
    margin: 0;
}

/* ── Metric badges ── */
.metric-row {
    display: flex;
    gap: 12px;
    margin-top: 16px;
    flex-wrap: wrap;
}
.metric-badge {
    background: rgba(66,153,225,0.12);
    border: 1px solid rgba(66,153,225,0.25);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.8rem;
    color: #90cdf4;
    font-weight: 500;
}

/* ── Source chip ── */
.source-chip {
    display: inline-block;
    background: rgba(72,187,120,0.1);
    border: 1px solid rgba(72,187,120,0.25);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.78rem;
    color: #68d391;
    margin: 3px 3px 3px 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1528 0%, #0a0e1a 100%);
    border-right: 1px solid rgba(99,179,237,0.1);
}

/* ── Example button ── */
.stButton > button {
    width: 100%;
    background: rgba(255,255,255,0.04) !important;
    color: #90cdf4 !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    border-radius: 10px !important;
    text-align: left !important;
    font-size: 0.85rem !important;
    padding: 10px 14px !important;
    transition: all 0.2s !important;
    white-space: normal !important;
    height: auto !important;
}
.stButton > button:hover {
    background: rgba(99,179,237,0.1) !important;
    border-color: rgba(99,179,237,0.4) !important;
    transform: translateX(3px) !important;
}

/* ── Text area ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(99,179,237,0.25) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: rgba(99,179,237,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.1) !important;
}

/* ── Submit button override ── */
div[data-testid="stForm"] .stButton > button {
    background: linear-gradient(135deg, #2b6cb0, #3182ce) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 14px 28px !important;
    border-radius: 12px !important;
    letter-spacing: 0.3px !important;
}
div[data-testid="stForm"] .stButton > button:hover {
    background: linear-gradient(135deg, #3182ce, #4299e1) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(66,153,225,0.35) !important;
}

/* ── Status dot ── */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #48bb78;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.4; }
}
.status-text { color: #68d391; font-size: 0.85rem; font-weight: 500; }

/* ── Code blocks inside answers ── */
code { font-family: 'Fira Code', monospace !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Session state init ───────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []       # list of {question, answer, sources, latency, ts}
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "api_ok" not in st.session_state:
    st.session_state.api_ok = False

# ─── Helper: call API ─────────────────────────────────────────────────────────
def ask_api(question: str) -> dict | None:
    try:
        t0 = time.time()
        resp = requests.post(
            f"{API_BASE}/ask",
            json={"question": question},
            timeout=60,
        )
        elapsed = time.time() - t0
        if resp.status_code == 200:
            data = resp.json()
            data["latency"] = round(elapsed, 2)
            return data
        else:
            st.error(f"API error {resp.status_code}: {resp.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI backend. Make sure `uvicorn main:app --port 8000` is running.")
        return None

def check_health() -> dict | None:
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🐍 Python Q&A Assistant")
    st.markdown("---")

    # Health check
    health = check_health()
    if health:
        st.session_state.api_ok = True
        st.markdown(
            f'<span class="status-dot"></span>'
            f'<span class="status-text">API Online</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"📚 {health['index_size']:,} Q&As indexed")
        st.caption(f"⏱️ Uptime: {health['uptime_seconds']:.0f}s")
        st.caption(f"🔢 Requests: {health['requests_served']}")
    else:
        st.markdown("🔴 API Offline")
        st.warning("Start the API:\n```\nuvicorn main:app --port 8000\n```")

    st.markdown("---")
    st.markdown("#### 💡 Example Questions")

    EXAMPLES = [
        "How do I reverse a list in Python?",
        "What is a Python decorator?",
        "How do I handle exceptions in Python?",
        "What is the difference between list and tuple?",
        "How do I sort a dictionary by value?",
        "How do I use *args and **kwargs?",
        "What does the yield keyword do?",
        "How do I read a CSV file with pandas?",
    ]
    for ex in EXAMPLES:
        if st.button(ex, key=f"ex_{ex[:20]}"):
            st.session_state.current_question = ex

    st.markdown("---")

    # History
    if st.session_state.history:
        st.markdown("#### 🕘 Recent Questions")
        for i, h in enumerate(reversed(st.session_state.history[-8:])):
            st.markdown(
                f'<div class="history-item">'
                f'<p class="history-q">{h["question"][:60]}{"..." if len(h["question"]) > 60 else ""}</p>'
                f'<p class="history-time">{h["ts"]} · {h["latency"]}s</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if st.session_state.history:
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

# ─── Main content ─────────────────────────────────────────────────────────────
# Hero
st.markdown("""
<div class="hero">
    <h1 class="hero-title">🐍 Python Q&A Assistant</h1>
    <p class="hero-subtitle">
        Powered by <strong>Stack Overflow</strong> data · <strong>RAG Pipeline</strong> ·
        <strong>Groq LLaMA-3.3-70b</strong> · 14,864 real Q&A pairs
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Question input form ───────────────────────────────────────────────────────
with st.form("question_form", clear_on_submit=False):
    question = st.text_area(
        "Ask a Python question",
        value=st.session_state.current_question,
        placeholder="e.g. How do I reverse a list in Python? What is a lambda function? How does GIL work?",
        height=110,
        label_visibility="collapsed",
    )
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        submitted = st.form_submit_button("🔍  Ask Question", use_container_width=True)
    with col2:
        st.form_submit_button("", disabled=True, use_container_width=True)  # spacer
    with col3:
        pass

# ─── Handle submission ────────────────────────────────────────────────────────
if submitted and question.strip():
    st.session_state.current_question = question.strip()

    with st.spinner("🔍 Searching Stack Overflow knowledge base..."):
        result = ask_api(question.strip())

    if result:
        entry = {
            "question": question.strip(),
            "answer":   result["answer"],
            "sources":  result.get("sources", []),
            "latency":  result.get("latency", 0),
            "model":    result.get("model", ""),
            "docs":     result.get("docs_retrieved", 5),
            "ts":       datetime.now().strftime("%H:%M"),
        }
        st.session_state.history.append(entry)

elif submitted and not question.strip():
    st.warning("Please type a question first!")

# ─── Display latest answer ────────────────────────────────────────────────────
if st.session_state.history:
    latest = st.session_state.history[-1]

    st.markdown("---")

    # Question bubble
    st.markdown(
        f"""
        <div style="
            background: rgba(66,153,225,0.1);
            border: 1px solid rgba(66,153,225,0.2);
            border-radius: 14px 14px 4px 14px;
            padding: 14px 20px;
            margin-bottom: 4px;
            display: inline-block;
            max-width: 80%;
            float: right;
            clear: both;
        ">
            <span style="color:#90cdf4; font-weight:600; font-size:0.95rem;">
                {latest['question']}
            </span>
        </div>
        <div style="clear:both; margin-bottom:16px;"></div>
        """,
        unsafe_allow_html=True,
    )

    # Answer card
    st.markdown('<div class="answer-card">', unsafe_allow_html=True)

    # Metrics row
    sources_html = "".join(
        f'<span class="source-chip">📎 {s[:55]}{"..." if len(s)>55 else ""}</span>'
        for s in latest["sources"]
    )
    st.markdown(
        f"""
        <div class="metric-row">
            <span class="metric-badge">⚡ {latest['latency']}s</span>
            <span class="metric-badge">🤖 {latest['model'].split('-')[0]+'-'+latest['model'].split('-')[1] if '-' in latest['model'] else latest['model']}</span>
            <span class="metric-badge">📚 {latest['docs']} sources</span>
            <span class="metric-badge">🕐 {latest['ts']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Answer text (rendered as markdown for code highlighting)
    st.markdown(latest["answer"])

    # Sources expander
    if latest["sources"]:
        with st.expander("📎 View Stack Overflow Sources", expanded=False):
            for i, src in enumerate(latest["sources"], 1):
                st.markdown(
                    f'<span class="source-chip">#{i} {src}</span>',
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Previous Q&As ──────────────────────────────────────────────────────────
    if len(st.session_state.history) > 1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📜 Previous Answers")
        for h in reversed(st.session_state.history[:-1]):
            with st.expander(f"Q: {h['question'][:80]}...  ({h['ts']} · {h['latency']}s)"):
                st.markdown(h["answer"])
                if h["sources"]:
                    srcs = " ".join(
                        f'<span class="source-chip">📎 {s[:50]}</span>'
                        for s in h["sources"]
                    )
                    st.markdown(srcs, unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    cards = [
        ("🔍", "RAG Powered", "Retrieves real Stack Overflow answers as context before generating"),
        ("⚡", "Groq LLaMA 3.3", "Blazing fast inference with the best open-source LLM"),
        ("📚", "14,864 Q&As", "Indexed from 607K Python Stack Overflow questions"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(
                f"""
                <div style="
                    background: rgba(255,255,255,0.03);
                    border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 14px;
                    padding: 24px 20px;
                    text-align: center;
                    height: 160px;
                ">
                    <div style="font-size:2rem; margin-bottom:10px;">{icon}</div>
                    <div style="color:#90cdf4; font-weight:600; font-size:0.95rem; margin-bottom:6px;">{title}</div>
                    <div style="color:#64748b; font-size:0.82rem; line-height:1.5;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
