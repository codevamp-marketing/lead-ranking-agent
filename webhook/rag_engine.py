"""
webhook/rag_engine.py — RAG Engine for Invertis University Course Q&A
=======================================================================
Pipeline:
  1. Keyword gate     → is_course_query() — word-boundary regex
  2. Embedding        → all-MiniLM-L6-v2 (local, free, 384-dim)
  3. Vector retrieval → pgvector via Supabase RPC match_course_chunks
  4. History          → last 6 turns from ConversationMessage table
  5. LLM generation   → Groq llama-3.1-8b-instant
                        Falls back to formatted chunks if Groq unreachable

FIXES in this version
─────────────────────
FIX 1 — Wrong course in follow-up questions (the core RAG issue).
         Root cause: "Whats the eligibility criteria of that course"
         contains no course name, so vector search finds unrelated course.
         Fix: inject last mentioned course name into the query before
         embedding — called "query enrichment". Now "eligibility of that
         course" becomes "eligibility MBA" and retrieves the right chunks.

FIX 2 — Conversation history now extracts last_course_mentioned.
         Scans the last 6 messages for any course keyword and passes it
         to query enrichment so follow-up questions stay on-topic.

FIX 3 — System prompt now explicitly tells LLM to maintain topic continuity
         and references "the course being discussed" in follow-ups.

FIX 4 — Removed debug print statements (===RAW DB RESULT===) that
         polluted production logs.

FIX 5 — Groq client timeout set to 15s to prevent indefinite hangs.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Optional

from sentence_transformers import SentenceTransformer

from webhook.conversation_store import get_conversation_history, _get_supabase
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Shared constant — MUST match ingest_courses.py ───────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384

# ── Embedding model — loaded once at process start ────────────────────────────
_embedder = SentenceTransformer(EMBEDDING_MODEL)
logger.info("rag_models_loaded", embedding_model=EMBEDDING_MODEL)

# ── Similarity threshold ──────────────────────────────────────────────────────
_MIN_SIMILARITY = 0.25


# ═══════════════════════════════════════════════════════════════════════════════
#  KEYWORD GATE — single definition, word-boundary regex
# ═══════════════════════════════════════════════════════════════════════════════

_COURSE_KEYWORDS: set[str] = {
    # Fee / cost
    "fee", "fees", "cost", "costs", "price", "pricing", "tuition", "charges",
    "payment", "scholarship", "scholarships", "stipend",
    # Course names
    "course", "courses", "programme", "programs", "degree", "degrees",
    "btech", "b.tech", "mtech", "m.tech", "mba", "mca", "pgdm",
    "bca", "bba", "msc", "m.sc", "phd", "ph.d", "diploma",
    "engineering", "management", "law", "agriculture", "pharmacy",
    "bsc", "b.sc", "computer", "science", "mechanical", "civil",
    "electrical", "electronics", "biotechnology", "forensic",
    # Admission
    "admission", "admissions", "apply", "application", "enroll",
    "eligibility", "eligible", "qualify", "qualification", "criteria",
    "entrance", "merit", "cutoff",
    # Academic
    "syllabus", "curriculum", "subjects", "semester", "duration",
    "intake", "seats", "batch", "placement", "placements", "internship",
    # Campus
    "hostel", "accommodation", "campus", "facilities", "library",
    "laboratory", "lab", "sports",
    # Intent words
    "detail", "details", "information", "structure", "tell me",
    "how much", "what is", "what are",
}

_KW_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in _COURSE_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def is_course_query(text: str) -> bool:
    return bool(_KW_PATTERN.search(text))


# ── Course name patterns for context extraction ───────────────────────────────
# Used by _extract_last_course() to find what course was last discussed.
_COURSE_NAME_PATTERNS = [
    r"\bMBA\b", r"\bMCA\b", r"\bBCA\b", r"\bBBA\b", r"\bPGDM\b",
    r"\bB\.?Tech\b", r"\bM\.?Tech\b", r"\bB\.?Sc\.?\b", r"\bM\.?Sc\.?\b",
    r"\bPh\.?D\.?\b", r"\bLL\.?B\b", r"\bLL\.?M\b",
    r"\bDiploma\b", r"\bEngineering\b", r"\bManagement\b",
    r"\bForensic\b", r"\bBiotechnology\b", r"\bComputer Science\b",
    r"\bMechanical\b", r"\bCivil\b", r"\bElectrical\b",
    r"\bAgriculture\b", r"\bPharmacy\b", r"\bCommerce\b",
    r"\bFinTech\b", r"\bArtificial Intelligence\b",
]
_COURSE_PATTERN = re.compile(
    "|".join(_COURSE_NAME_PATTERNS), re.IGNORECASE
)


def _extract_last_course(history: list[dict]) -> str:
    """
    FIX 1 & 2: Scans recent conversation history (newest first) and
    returns the last course name that was mentioned.
    Used to enrich vague follow-up queries like "eligibility of that course".
    """
    # Scan messages newest-first
    for msg in reversed(history):
        body = msg.get("body", "")
        match = _COURSE_PATTERN.search(body)
        if match:
            return match.group(0)
    return ""


def _enrich_query(user_message: str, last_course: str) -> str:
    """
    FIX 1: If the query is vague (short, no course name) and we know
    the last discussed course, inject it into the query before embedding.

    Examples:
      "eligibility of that course" + last_course="MBA"
      → "eligibility of that course MBA"

      "what are the fees?" + last_course="B.Tech"
      → "what are the fees? B.Tech"

      "what is MBA?" + last_course="MBA"
      → "what is MBA?"  (already has course name, no enrichment needed)
    """
    if not last_course:
        return user_message

    # Only enrich if message doesn't already mention the course
    if last_course.lower() in user_message.lower():
        return user_message

    # Only enrich short/vague messages (longer messages are likely self-contained)
    if len(user_message.split()) > 12:
        return user_message

    enriched = f"{user_message} {last_course}"
    logger.info("query_enriched", original=user_message[:60], enriched=enriched[:80])
    return enriched


# ═══════════════════════════════════════════════════════════════════════════════
#  GROQ CLIENT — lazy init with timeout
# ═══════════════════════════════════════════════════════════════════════════════

_groq_client = None


def _get_groq():
    global _groq_client
    if _groq_client is None:
        try:
            import httpx
            from groq import Groq
            api_key = os.environ.get("GROQ_API_KEY", "")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY not set in .env")
            _groq_client = Groq(
                api_key     = api_key,
                http_client = httpx.Client(timeout=httpx.Timeout(15.0)),
            )
        except ImportError:
            raise RuntimeError("groq not installed — run: pip install groq")
    return _groq_client


# ═══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = _SYSTEM_PROMPT = """You are a helpful admissions assistant at Invertis University, Bareilly.
Answer students' questions about courses, fees, eligibility, hostel, and campus life.

STRICT RULES:
1. Answer ONLY from the COURSE INFORMATION block provided below.
2. If the answer is not in the course information, say:
   "I don't have that specific detail right now — our counsellor will share it shortly. 🙏"
3. NEVER invent fees, dates, eligibility marks, or course names.
4. Always reply in English unless the student writes in Hindi.
5. Do NOT use the student's name anywhere in the reply.
6. Do NOT write in paragraph form. Always use this exact format:

---
[One line brief intro sentence about the course or topic asked]

*Key Details:*
- [Point 1]
- [Point 2]
- [Point 3]
- [Point 4 if available]

[One closing line offering further help]

— Invertis University Admissions
---

7. Keep each bullet point short and factual — one piece of information per point.
8. The closing line should always offer something specific like fees, eligibility, hostel, or admission process."""

# ═══════════════════════════════════════════════════════════════════════════════
#  CHUNK FORMATTER — used as LLM fallback
# ═══════════════════════════════════════════════════════════════════════════════

def _format_chunks_as_reply(lead_name: str, chunks: list[dict]) -> str:
    if not chunks:
       return (
            "I don't have specific details on that right now. "
            "Our counsellor will reach out with accurate information shortly. 🙏"
        )
    lines =  " Here's what I found:\n"
    for chunk in chunks[:2]:
        course = chunk.get("course_name", "")
        text   = chunk.get("chunk_text", "").strip()
        if course:
            lines.append(f"*{course}*")
        lines.append(text[:400])
        lines.append("")
    lines.append("Would you like more details or to speak with a counsellor? 😊")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  VECTOR SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def _search_courses_sync(
    query_embedding: list[float],
    top_k: int = 4,
) -> tuple[str, list[dict]]:
    """
    Calls match_course_chunks RPC. Returns (formatted_context, raw_chunks).
    FIX 4: Removed debug print statements.
    """
    try:
        db     = _get_supabase()
        result = db.rpc("match_course_chunks", {
            "query_embedding": query_embedding,
            "match_count":     top_k,
        }).execute()

        chunks = result.data or []

        if not chunks:
            logger.info("vector_search_empty")
            return "", []

        good_chunks = [
            r for r in chunks
            if float(r.get("similarity", 0)) >= _MIN_SIMILARITY
        ]

        if not good_chunks:
            best = max(float(r.get("similarity", 0)) for r in chunks)
            logger.info(
                "vector_search_below_threshold",
                best=round(best, 3),
                threshold=_MIN_SIMILARITY,
            )
            return "", []

        lines = []
        for r in good_chunks:
            course = r.get("course_name", "Course")
            sim    = float(r.get("similarity", 0))
            text   = r.get("chunk_text", "")
            lines.append(f"[{course}] (relevance: {sim:.2f})\n{text}")

        logger.info("vector_search_ok", chunks=len(good_chunks),
                    top_course=good_chunks[0].get("course_name"),
                    top_sim=round(float(good_chunks[0].get("similarity", 0)), 3))
        return "\n\n---\n\n".join(lines), good_chunks

    except Exception as exc:
        logger.warning("vector_search_failed", error=str(exc))
        return "", []


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_sync(
    lead_name:    str,
    user_message: str,
    context:      str,
    chat_history: str,
    raw_chunks:   list[dict],
    last_course:  str,
) -> str:
    """
    FIX 3: Passes last_course to the LLM so it knows which course
    is being discussed in follow-up questions.
    """
    sections: list[str] = [
        f"=== COURSE INFORMATION ===\n{context}\n=== END ===",
    ]

    if chat_history.strip():
        sections.append(
            f"=== RECENT CONVERSATION ===\n{chat_history}\n=== END ==="
        )

    # FIX 3: explicitly tell LLM what course is being discussed
    if last_course:
        sections.append(
            f"NOTE: The student is currently asking about: {last_course}. "
            "Use this to resolve references like 'that course' or 'the same course'."
        )

    sections.append(f"Student ({lead_name}): {user_message}")
    sections.append("Counsellor (2-4 sentences, English only):")
    user_turn = "\n\n".join(sections)

    try:
        groq     = _get_groq()
        response = groq.chat.completions.create(
            model       = "llama-3.1-8b-instant",
            max_tokens  = 500,
            temperature = 0.3,   # lower = more factual
            messages    = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_turn},
            ],
        )
        reply = response.choices[0].message.content.strip()
        logger.info("llm_ok", tokens=response.usage.total_tokens)
        return reply

    except Exception as exc:
        logger.error("groq_failed", error=str(exc))
        return _format_chunks_as_reply(raw_chunks)


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

async def rag_reply(
    lead_phone:   str,
    user_message: str,
    lead_name:    str,
) -> str:
    """
    Full RAG pipeline. Never raises — always returns a reply string.

    Steps:
      1. Fetch conversation history (needed for query enrichment)
      2. Extract last course mentioned (FIX 1 & 2)
      3. Enrich vague queries with course context (FIX 1)
      4. Embed enriched query
      5. Vector search
      6. Safe fallback if no relevant chunks
      7. Generate via Groq with course context (FIX 3)
    """
    # Step 1 — Fetch history first (needed for query enrichment)
    try:
        history = await get_conversation_history(lead_phone, limit=6)
        chat_history = "\n".join(
            f"{'Student' if m['direction'] == 'inbound' else 'Counsellor'}: "
            f"{m['body'][:200]}"
            for m in history
        )
    except Exception as exc:
        logger.warning("history_fetch_failed", error=str(exc))
        history      = []
        chat_history = ""

    # Step 2 — Extract last mentioned course from history + current message
    all_messages = history + [{"body": user_message, "direction": "inbound"}]
    last_course  = _extract_last_course(all_messages)

    # Step 3 — Enrich vague query (FIX 1 — prevents wrong-course replies)
    enriched_query = _enrich_query(user_message, last_course)

    # Step 4 — Embed enriched query
    try:
        q_vec: list[float] = await asyncio.to_thread(
            lambda: _embedder.encode(enriched_query).tolist()
        )
    except Exception as exc:
        logger.error("embedding_failed", error=str(exc))
        return "Hi Our counsellor will reach out with details shortly. 🙏"

    # Step 5 — Vector search
    context, raw_chunks = await asyncio.to_thread(
        _search_courses_sync, q_vec, 4
    )

    # Step 6 — Safe fallback if no relevant chunks
    if not context:
        logger.info("rag_no_context", query=enriched_query[:80])
        return (
            "Hi  I don't have specific details on that right now. "
            "Our counsellor will reach out with accurate information shortly. 🙏\n\n"
            "You can also ask me about course fees, eligibility, or the admission process! 😊"
        )

    # Step 7 — Generate with course context (FIX 3)
    reply = await asyncio.to_thread(
        _generate_sync,
        lead_name, user_message, context, chat_history, raw_chunks, last_course,
    )

    logger.info(
        "rag_reply_generated",
        lead_phone    = lead_phone,
        query         = user_message[:60],
        enriched      = enriched_query[:60],
        last_course   = last_course or "none",
        context_len   = len(context),
    )
    return reply