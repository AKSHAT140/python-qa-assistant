"""
preprocess.py
-------------
Step 1: Filter Python-tagged questions, join best answers, clean HTML.
Output: data/clean_qa.parquet  (~15K top-scored Python Q&A pairs)

Run once locally before ingesting:
    python preprocess.py
"""

import pandas as pd
from bs4 import BeautifulSoup
import re
import os

DATASET_PATH = "dataset/archive"
OUTPUT_PATH  = "data/clean_qa.parquet"
TOP_N        = 15000   # Keep top N questions by score (keeps file sizes manageable)

# ─── 1. Load Tags → get Python question IDs ──────────────────────────────────
print("📂 Loading Tags.csv...")
tags = pd.read_csv(f"{DATASET_PATH}/Tags.csv", encoding="latin-1")
python_ids = set(tags[tags["Tag"] == "python"]["Id"].tolist())
print(f"   ✅ {len(python_ids):,} Python-tagged question IDs found")

# ─── 2. Load Questions in chunks (file is ~888MB) ────────────────────────────
print("\n📂 Loading Questions.csv (chunked, this may take a minute)...")
q_chunks = []
for chunk in pd.read_csv(
    f"{DATASET_PATH}/Questions.csv",
    encoding="latin-1",
    chunksize=50_000,
    on_bad_lines="skip",
):
    filtered = chunk[chunk["Id"].isin(python_ids)]
    if not filtered.empty:
        q_chunks.append(filtered)

questions = pd.concat(q_chunks, ignore_index=True)
print(f"   ✅ {len(questions):,} Python questions loaded")

# Keep only top N by vote score
questions = questions.sort_values("Score", ascending=False).head(TOP_N)
print(f"   ✅ Trimmed to top {len(questions):,} by score")

# ─── 3. Load Answers for only those questions ────────────────────────────────
print("\n📂 Loading Answers.csv (chunked, this may take a minute)...")
target_ids = set(questions["Id"].tolist())
a_chunks = []
for chunk in pd.read_csv(
    f"{DATASET_PATH}/Answers.csv",
    encoding="latin-1",
    chunksize=50_000,
    on_bad_lines="skip",
):
    filtered = chunk[chunk["ParentId"].isin(target_ids)]
    if not filtered.empty:
        a_chunks.append(filtered)

answers = pd.concat(a_chunks, ignore_index=True)
print(f"   ✅ {len(answers):,} answers loaded")

# ─── 4. Pick best answer per question (highest score) ────────────────────────
best_answers = (
    answers.sort_values("Score", ascending=False)
    .groupby("ParentId", as_index=False)
    .first()
)
best_answers = best_answers.rename(
    columns={"ParentId": "QuestionId", "Body": "AnswerBody", "Score": "AnswerScore"}
)
print(f"   ✅ {len(best_answers):,} best answers selected")

# ─── 5. Clean HTML ────────────────────────────────────────────────────────────
def clean_html(html: str) -> str:
    """Strip HTML tags, preserve code blocks with backtick markers."""
    if pd.isna(html):
        return ""
    soup = BeautifulSoup(str(html), "html.parser")
    # Mark code blocks so they stay readable
    for code in soup.find_all("code"):
        code.string = f"\n```\n{code.get_text()}\n```\n"
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("\n🧹 Cleaning HTML bodies...")
questions["CleanBody"]    = questions["Body"].apply(clean_html)
best_answers["CleanAnswer"] = best_answers["AnswerBody"].apply(clean_html)

# ─── 6. Merge questions + best answers ───────────────────────────────────────
merged = questions.merge(
    best_answers[["QuestionId", "CleanAnswer", "AnswerScore"]],
    left_on="Id",
    right_on="QuestionId",
    how="inner",
)

merged = merged[[
    "Id", "Title", "CleanBody", "Score", "CleanAnswer", "AnswerScore"
]].copy()
merged.columns = [
    "question_id", "title", "question_body", "question_score",
    "answer_body", "answer_score"
]

# ─── 7. Basic quality filters ─────────────────────────────────────────────────
merged = merged[merged["answer_body"].str.len() > 50]
merged = merged[merged["question_body"].str.len() > 20]
merged = merged[merged["title"].str.len() > 5]
merged = merged.drop_duplicates(subset="question_id")

print(f"\n✅ Final dataset: {len(merged):,} Q&A pairs")

# ─── 8. Save ──────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
merged.to_parquet(OUTPUT_PATH, index=False)
print(f"💾 Saved → {OUTPUT_PATH}")

print("\nSample row:")
print(merged.iloc[0][["title", "question_body", "answer_body"]].to_string())
