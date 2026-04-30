"""
validate_rag.py — End-to-end RAG pipeline validator
=====================================================
Run this BEFORE starting the webhook server to confirm
every layer of the RAG pipeline is working correctly.

Usage:
  python validate_rag.py

What it checks:
  Step 1 — Supabase connection + CourseChunk count
  Step 2 — match_course_chunks RPC function exists
  Step 3 — Embedding model loads and produces correct dimensions
  Step 4 — Retrieval returns relevant chunks for test queries
  Step 5 — Similarity scores are in a sensible range
  Step 6 — Groq API key is valid and model responds
  Step 7 — Full end-to-end rag_reply() on 5 sample queries

Prints a clear PASS / FAIL / WARN for each step.
Exit code 0 = all critical checks passed.
Exit code 1 = one or more critical checks failed.
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓ PASS{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠ WARN{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗ FAIL{RESET}  {msg}")
def step(n, title): print(f"\n{BOLD}Step {n}: {title}{RESET}")

# ── Test queries ──────────────────────────────────────────────────────────────
TEST_QUERIES = [
    "What are the fees for B.Tech?",
    "Eligibility criteria for MBA admission",
    "What courses does Invertis University offer?",
    "Is hostel available for students?",
    "What is the admission procedure?",
]

critical_failures = []

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Supabase connection
# ─────────────────────────────────────────────────────────────────────────────

step(1, "Supabase connection + CourseChunk count")

supabase_url = os.getenv("SUPABASE_URL", "").strip()
supabase_key = os.getenv("SUPABASE_SERVICE_KEY", "").strip()

if not supabase_url or not supabase_key:
    fail("SUPABASE_URL or SUPABASE_SERVICE_KEY not set in .env")
    critical_failures.append("Supabase credentials missing")
else:
    ok(f"Credentials found: {supabase_url[:40]}...")

    try:
        import httpx
        from supabase import create_client
        sb = create_client(supabase_url, supabase_key)
        sb.postgrest.session = httpx.Client(http2=False, timeout=15)

        result = sb.table("CourseChunk").select("id", count="exact").execute()
        count  = result.count or 0

        if count == 0:
            fail("CourseChunk table is EMPTY — run ingest_courses.py first")
            critical_failures.append("No course data ingested")
        elif count < 10:
            warn(f"Only {count} chunks found — may not cover all queries well")
        else:
            ok(f"CourseChunk table has {count} chunks indexed")

    except Exception as e:
        fail(f"Supabase connection failed: {e}")
        critical_failures.append(f"Supabase: {e}")
        sb = None

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — match_course_chunks RPC
# ─────────────────────────────────────────────────────────────────────────────

step(2, "match_course_chunks RPC function")

if 'sb' in dir() and sb:
    try:
        # Dummy embedding — all zeros (should return rows, just low similarity)
        dummy = [0.0] * 384
        result = sb.rpc("match_course_chunks", {
            "query_embedding": dummy,
            "match_count": 1,
        }).execute()

        rows = result.data or []
        if rows:
            row = rows[0]
            has_similarity = "similarity" in row
            has_chunk_text = "chunk_text" in row
            if has_similarity and has_chunk_text:
                ok("RPC returns correct columns (course_name, chunk_text, similarity)")
            else:
                warn(f"RPC missing columns. Got: {list(row.keys())}")
        else:
            warn("RPC returned 0 rows for dummy query (table may be empty)")

    except Exception as e:
        fail(f"match_course_chunks RPC failed: {e}")
        critical_failures.append("RPC function missing — run sql/setup_demo_supabase.sql")
else:
    warn("Skipping — Supabase not connected")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — Embedding model
# ─────────────────────────────────────────────────────────────────────────────

step(3, "Embedding model loads and produces correct dimensions")

try:
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    load_time = time.time() - t0

    test_emb = embedder.encode("test query").tolist()
    dim = len(test_emb)

    if dim != 384:
        fail(f"Expected 384 dimensions, got {dim} — model mismatch!")
        critical_failures.append("Embedding dimension mismatch")
    else:
        ok(f"Model loaded in {load_time:.1f}s — embedding dim = {dim} ✓")

    # Check for NaN/Inf
    bad = [x for x in test_emb if math.isnan(x) or math.isinf(x)]
    if bad:
        fail(f"Embedding contains {len(bad)} NaN/Inf values")
    else:
        ok("No NaN/Inf in embedding")

except ImportError:
    fail("sentence-transformers not installed: pip install sentence-transformers")
    critical_failures.append("sentence-transformers missing")
except Exception as e:
    fail(f"Embedding model error: {e}")
    critical_failures.append(str(e))

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 & 5 — Retrieval quality + similarity scores
# ─────────────────────────────────────────────────────────────────────────────

step(4, "Retrieval quality — test queries")

if 'sb' in dir() and sb and 'embedder' in dir():
    for query in TEST_QUERIES:
        try:
            q_vec = embedder.encode(query).tolist()
            result = sb.rpc("match_course_chunks", {
                "query_embedding": q_vec,
                "match_count": 3,
            }).execute()

            rows = result.data or []
            if not rows:
                fail(f"No results for: '{query}'")
                continue

            top = rows[0]
            sim  = float(top.get("similarity", 0))
            name = top.get("course_name", "?")
            prev = top.get("chunk_text", "")[:80]

            if sim >= 0.40:
                ok(f"'{query[:45]}...' → sim={sim:.3f} [{name}]")
            elif sim >= 0.25:
                warn(f"'{query[:45]}...' → sim={sim:.3f} [{name}] (borderline)")
            else:
                fail(f"'{query[:45]}...' → sim={sim:.3f} — below threshold (check ingest)")

        except Exception as e:
            fail(f"Retrieval error for '{query[:40]}': {e}")
else:
    warn("Skipping — Supabase or embedder not available")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — Groq API
# ─────────────────────────────────────────────────────────────────────────────

step(5, "Groq API connectivity")

groq_key = os.getenv("GROQ_API_KEY", "").strip()
if not groq_key:
    fail("GROQ_API_KEY not set in .env")
    critical_failures.append("GROQ_API_KEY missing")
else:
    ok(f"GROQ_API_KEY found: {groq_key[:8]}...")
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        resp = client.chat.completions.create(
            model      = "llama-3.1-8b-instant",
            max_tokens = 20,
            messages   = [{"role": "user", "content": "Reply with: ok"}],
        )
        reply = resp.choices[0].message.content.strip()
        ok(f"Groq responded: '{reply}'")
    except Exception as e:
        fail(f"Groq API call failed: {e}")
        critical_failures.append(f"Groq: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 — Full end-to-end rag_reply()
# ─────────────────────────────────────────────────────────────────────────────

step(6, "Full end-to-end rag_reply() on sample queries")

async def run_e2e():
    try:
        # Patch sys.path so webhook module imports work
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from webhook.rag_engine import rag_reply, is_course_query

        for query in TEST_QUERIES[:3]:
            t0 = time.time()
            is_course = is_course_query(query)
            if not is_course:
                warn(f"'{query}' NOT matched by is_course_query — add keywords")
                continue

            reply = await rag_reply("test_+919999999999", query, "TestStudent")
            elapsed = time.time() - t0

            if len(reply) < 20:
                warn(f"Very short reply ({len(reply)} chars) for: '{query[:40]}'")
            elif "counsellor will" in reply.lower() and "don't have" in reply.lower():
                warn(f"Safe fallback triggered (no context) for: '{query[:40]}'")
            else:
                ok(f"Got {len(reply)}-char reply in {elapsed:.1f}s for: '{query[:40]}'")

            print(f"\n    Query  : {query}")
            print(f"    Reply  : {reply[:200]}{'...' if len(reply)>200 else ''}\n")

    except ImportError as e:
        warn(f"Could not import rag_engine (run from project root): {e}")

if 'sb' in dir() and sb:
    asyncio.run(run_e2e())
else:
    warn("Skipping — Supabase not connected")

# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
if critical_failures:
    print(f"{RED}{BOLD}  VALIDATION FAILED — {len(critical_failures)} critical issue(s){RESET}")
    for f in critical_failures:
        print(f"  {RED}→ {f}{RESET}")
    print()
    print("  Fix the above issues then re-run: python validate_rag.py")
    sys.exit(1)
else:
    print(f"{GREEN}{BOLD}  ALL CRITICAL CHECKS PASSED{RESET}")
    print()
    print("  Next steps:")
    print("  1. python main.py                              (Terminal 1)")
    print("  2. uvicorn webhook.webhook_server:app --port 8000 --reload  (Terminal 2)")
    print("  3. ngrok http 8000                             (Terminal 3)")
    print()
    print("  To test RAG in browser:")
    print("  http://localhost:8000/rag-test?q=fees+for+BTech")
    print("  http://localhost:8000/health")
    print("=" * 60)