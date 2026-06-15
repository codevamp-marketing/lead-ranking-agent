#!/usr/bin/env python3
"""
diagnose.py — Quick health check for the entire system.
Run this FIRST before starting any service.

  python diagnose.py

It checks everything in order and tells you exactly what's broken.
"""

from __future__ import annotations
import os
import sys

# ── 1. Load env ───────────────────────────────────────────────────────────────
from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
if not dotenv_path:
    print("❌ .env file NOT FOUND in this directory or any parent.")
    print("   Create .env in the ranking-agent/ folder.")
    sys.exit(1)
load_dotenv(dotenv_path, override=True)
print(f"✅ .env loaded from: {dotenv_path}")

# ── 2. Check required env vars ────────────────────────────────────────────────
REQUIRED = [
    "SUPABASE_URL", "SUPABASE_SERVICE_KEY", "DATABASE_URL",
    "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM",
    "ADMIN_PHONE", "WHATSAPP_MODE", "GROQ_API_KEY",
]
print("\n── ENV VARS ─────────────────────────────────────────────────")
all_ok = True
for key in REQUIRED:
    val = os.environ.get(key, "")
    if val:
        preview = val[:30] + "..." if len(val) > 30 else val
        # Mask sensitive values
        if any(s in key for s in ("KEY", "TOKEN", "SID")):
            preview = val[:8] + "..." + val[-4:]
        print(f"  ✅ {key:<30} = {preview}")
    else:
        print(f"  ❌ {key:<30} = MISSING")
        all_ok = False

if not all_ok:
    print("\n  Fix: add missing vars to .env and re-run.")
    sys.exit(1)

# ── 3. Test Supabase connection ───────────────────────────────────────────────
print("\n── SUPABASE ─────────────────────────────────────────────────")
try:
    from supabase import create_client
    url = os.environ["SUPABASE_URL"].strip().strip('"').strip("'")
    key = os.environ["SUPABASE_SERVICE_KEY"].strip().strip('"').strip("'")
    if not url.startswith("http"):
        print(f"  ❌ SUPABASE_URL doesn't start with http: '{url[:40]}'")
        sys.exit(1)
    db = create_client(url, key)
    r  = db.table("Lead").select("id").limit(1).execute()
    print(f"  ✅ Supabase connected. Lead table accessible.")
except Exception as e:
    print(f"  ❌ Supabase connection failed: {e}")
    print("     Check SUPABASE_URL and SUPABASE_SERVICE_KEY in .env")
    sys.exit(1)

# ── 4. Check CourseChunk ──────────────────────────────────────────────────────
print("\n── COURSE CHUNKS ────────────────────────────────────────────")
try:
    r = db.table("CourseChunk").select("id", count="exact").execute()
    count = r.count or 0
    if count == 0:
        print("  ❌ CourseChunk table is EMPTY!")
        print("     Run: python ingest_courses.py --clear")
        sys.exit(1)
    else:
        print(f"  ✅ CourseChunk has {count} rows.")

    # Check a sample row
    sample = db.table("CourseChunk").select("course_name, embedding, chunk_text").limit(3).execute()
    rows = sample.data or []
    print(f"  ✅ Sample courses: {[r.get('course_name') for r in rows]}")
    
    # Check embedding is a list, not a string
    if rows:
        emb = rows[0].get("embedding")
        if isinstance(emb, list):
            print(f"  ✅ Embedding format: list, dim={len(emb)}")
        elif isinstance(emb, str):
            print(f"  ❌ Embedding stored as STRING — pgvector won't work!")
            print("     Re-ingest: python ingest_courses.py --clear")
        elif emb is None:
            print("  ❌ Embedding is NULL — re-ingest required")
except Exception as e:
    print(f"  ❌ CourseChunk check failed: {e}")

# ── 5. Test RPC ───────────────────────────────────────────────────────────────
print("\n── VECTOR SEARCH RPC ────────────────────────────────────────")
try:
    from sentence_transformers import SentenceTransformer
    print("  Loading embedding model (all-MiniLM-L6-v2)...")
    model   = SentenceTransformer("all-MiniLM-L6-v2")
    vec     = model.encode("B.Tech fee structure eligibility").tolist()
    result  = db.rpc("match_course_chunks", {
        "query_embedding": vec,
        "match_count":     5,
    }).execute()
    rows = result.data or []
    if rows:
        print(f"  ✅ RPC returned {len(rows)} results:")
        for r in rows:
            print(f"     [{r.get('similarity', 0):.3f}] {r.get('course_name')}")
    else:
        print("  ❌ RPC returned 0 results!")
        print("     This means match_course_chunks() SQL function doesn't exist,")
        print("     OR embeddings are stored as wrong type.")
        print("     Fix: run complete_schema_final.sql in Supabase SQL Editor.")
        print("     Then: python ingest_courses.py --clear")
except Exception as e:
    print(f"  ❌ RPC test failed: {e}")
    print("     The match_course_chunks function may be missing.")
    print("     Run complete_schema_final.sql in Supabase SQL Editor.")

# ── 6. Test ILIKE fallback ────────────────────────────────────────────────────
print("\n── DIRECT ILIKE FALLBACK ────────────────────────────────────")
try:
    for course in ["B.Tech", "BCA", "MBA"]:
        r = db.table("CourseChunk").select("course_name, chunk_text").ilike("course_name", f"%{course}%").limit(2).execute()
        rows = r.data or []
        if rows:
            print(f"  ✅ ILIKE '{course}' → {len(rows)} rows: {[x.get('course_name') for x in rows]}")
        else:
            print(f"  ❌ ILIKE '{course}' → 0 rows (course not in DB)")
except Exception as e:
    print(f"  ❌ ILIKE test failed: {e}")

# ── 7. Test Groq ──────────────────────────────────────────────────────────────
print("\n── GROQ API ─────────────────────────────────────────────────")
try:
    import httpx
    from groq import Groq
    client = Groq(
        api_key     = os.environ["GROQ_API_KEY"],
        http_client = httpx.Client(timeout=httpx.Timeout(10.0)),
    )
    resp = client.chat.completions.create(
        model      = "llama-3.1-8b-instant",
        max_tokens = 20,
        messages   = [{"role": "user", "content": "Say OK"}],
    )
    print(f"  ✅ Groq OK — response: '{resp.choices[0].message.content.strip()}'")
except Exception as e:
    print(f"  ❌ Groq failed: {e}")
    print("     Check GROQ_API_KEY in .env")

# ── 8. Test Twilio ────────────────────────────────────────────────────────────
print("\n── TWILIO ───────────────────────────────────────────────────")
try:
    from twilio.rest import Client as TwilioClient
    tc  = TwilioClient(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    acc = tc.api.accounts(os.environ["TWILIO_ACCOUNT_SID"]).fetch()
    print(f"  ✅ Twilio connected. Account: {acc.friendly_name}")
    print(f"  ℹ️  WHATSAPP_MODE = {os.environ.get('WHATSAPP_MODE')}")
    print(f"  ℹ️  WHATSAPP_FROM = {os.environ.get('TWILIO_WHATSAPP_FROM')}")
    print(f"  ℹ️  ADMIN_PHONE   = {os.environ.get('ADMIN_PHONE')}")
except Exception as e:
    print(f"  ❌ Twilio failed: {e}")

# ── 9. Test full RAG pipeline ─────────────────────────────────────────────────
print("\n── FULL RAG PIPELINE TEST ───────────────────────────────────")
try:
    import asyncio
    sys.path.insert(0, ".")
    from webhook.rag_engine import rag_reply, is_course_query

    test_msgs = [
        "i wanna know about the b tech fee structure",
        "what are the courses in mba",
        "tell me about bca admission",
    ]
    for msg in test_msgs:
        is_course = is_course_query(msg)
        print(f"\n  Query: '{msg}'")
        print(f"  is_course_query: {is_course}")
        if is_course:
            try:
                reply = asyncio.run(rag_reply("+919999999999", msg, "TestUser"))
                preview = reply[:150].replace("\n", " ")
                status = "✅" if "don't have" not in reply[:50] else "⚠️ "
                print(f"  {status} Reply: {preview}...")
            except Exception as e:
                print(f"  ❌ rag_reply failed: {e}")
        else:
            print("  ⚠️  Not routed to RAG — check is_course_query()")
except Exception as e:
    print(f"  ❌ RAG pipeline test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 62)
print("  DIAGNOSIS COMPLETE")
print("=" * 62)
print()
print("  If all ✅ above: start services with:")
print("  Terminal 1: python main.py")
print("  Terminal 2: uvicorn webhook.webhook_server:app --port 8000 --reload")
print("  Terminal 3: ngrok http 8000")
print()