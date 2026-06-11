"""
test_queries.py
---------------
Runs 10 test queries against the running API and saves results.

Usage:
    # Make sure the API is running first:
    #   uvicorn main:app --port 8000
    python test_queries.py
    
Output:
    test_results.json  — detailed results for each query
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

# ─── 10 diverse Python questions covering different topics ────────────────────
TEST_QUESTIONS = [
    # Basics
    "How do I reverse a list in Python?",
    "What is the difference between a list and a tuple in Python?",

    # Data structures
    "How do I sort a dictionary by value in Python?",
    "How do I merge two dictionaries in Python?",
    "How do I remove duplicates from a list while preserving order?",

    # OOP & advanced
    "What is a Python decorator and how do I use it?",
    "What is the difference between __str__ and __repr__ in Python?",
    "How do I use Python's *args and **kwargs?",

    # Data science / libraries
    "How do I read a CSV file using pandas in Python?",
    "How do I make an HTTP GET request in Python using the requests library?",
]

# ─── Health check ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  Python Q&A Assistant — API Test Suite")
print("=" * 65)

try:
    health = requests.get(f"{BASE_URL}/health", timeout=10)
    health.raise_for_status()
    h = health.json()
    print(f"\n✅ Health check passed")
    print(f"   Status:  {h['status']}")
    print(f"   Uptime:  {h['uptime_seconds']:.1f}s")
    print(f"   Index:   {h['index_size']:,} documents")
except Exception as e:
    print(f"❌ API not reachable at {BASE_URL}: {e}")
    print("   Make sure to run:  uvicorn main:app --port 8000")
    sys.exit(1)

# ─── Run tests ────────────────────────────────────────────────────────────────
results = []
print(f"\n🧪 Running {len(TEST_QUESTIONS)} test queries...\n")

for i, question in enumerate(TEST_QUESTIONS, 1):
    print(f"[{i:02d}/{len(TEST_QUESTIONS)}] {question}")
    t0 = time.time()

    try:
        resp = requests.post(
            f"{BASE_URL}/ask",
            json={"question": question},
            timeout=30,
        )
        elapsed = time.time() - t0

        if resp.status_code == 200:
            data = resp.json()
            print(f"       ✅ {elapsed:.2f}s | Sources: {data['sources'][:2]}")
            print(f"       → {data['answer'][:180].strip()}...\n")
            results.append({
                "test_id":        i,
                "question":       question,
                "answer":         data["answer"],
                "sources":        data["sources"],
                "model":          data["model"],
                "docs_retrieved": data["docs_retrieved"],
                "latency_s":      round(elapsed, 3),
                "status":         "PASS",
                "observation":    "Answer returned successfully with sources.",
            })
        else:
            print(f"       ❌ HTTP {resp.status_code}: {resp.text[:100]}\n")
            results.append({
                "test_id":     i,
                "question":    question,
                "status":      "FAIL",
                "http_status": resp.status_code,
                "error":       resp.text,
                "latency_s":   round(elapsed, 3),
                "observation": f"Unexpected HTTP status {resp.status_code}.",
            })

    except Exception as e:
        elapsed = time.time() - t0
        print(f"       ❌ Exception: {e}\n")
        results.append({
            "test_id":     i,
            "question":    question,
            "status":      "ERROR",
            "error":       str(e),
            "latency_s":   round(elapsed, 3),
            "observation": "Network or server error.",
        })

# ─── Summary ──────────────────────────────────────────────────────────────────
passed  = sum(1 for r in results if r["status"] == "PASS")
failed  = len(results) - passed
avg_lat = sum(r.get("latency_s", 0) for r in results if r["status"] == "PASS") / max(passed, 1)

print("=" * 65)
print(f"  Results: {passed}/{len(results)} passed  |  Avg latency: {avg_lat:.2f}s")
print("=" * 65)

if failed:
    print(f"\n⚠️  {failed} test(s) failed. Check test_results.json for details.")

# ─── Save results ─────────────────────────────────────────────────────────────
output = {
    "summary": {
        "total":       len(results),
        "passed":      passed,
        "failed":      failed,
        "avg_latency": round(avg_lat, 3),
    },
    "tests": results,
}

with open("test_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n💾 Full results saved → test_results.json")
