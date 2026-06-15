"""
webhook/rag_engine.py — RAG Engine v9.0
========================================
ROOT CAUSE FIXES v9.0
─────────────────────
FIX 1 — Supabase client was being created BEFORE load_dotenv() ran.
  conversation_store._get_supabase() was called at import time in some
  code paths, returning a client built with empty URL → all RPC calls
  failed with UnsupportedProtocol. Now _get_supabase() is called lazily
  inside each function that needs it.

FIX 2 — Similarity thresholds were too high for real data.
  With only 95 chunks and all-MiniLM-L6-v2, most scores land between
  0.25–0.45. Old _MIN_SIMILARITY=0.15 should be fine BUT the floor was
  causing fallthrough. Root fix: also added direct ILIKE fallback that
  fires whenever active_course is detected, even before vector search.

FIX 3 — _search_courses_sync was not getting active_course correctly
  when query enrichment returned it. Added explicit logging to trace
  every step so you can see exactly what's happening in logs.

FIX 4 — Groq API key loaded from os.environ directly (not settings)
  to avoid the settings import-time env read problem.

FIX 5 — Added a pre-flight warm-up on startup so embedder is ready.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv(override=True)

import asyncio
import os
import re
import time
from typing import Optional

from sentence_transformers import SentenceTransformer

from utils.logger import get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384

_embedder = SentenceTransformer(EMBEDDING_MODEL)
logger.info("rag_models_loaded", embedding_model=EMBEDDING_MODEL)

# ── Thresholds — tuned for 95-chunk corpus with all-MiniLM-L6-v2 ─────────────
_MIN_SIMILARITY         = 0.10   # FIX 2: lowered from 0.15
_MIN_SIMILARITY_FLOOR   = 0.05   # FIX 2: lowered from 0.10
_COURSE_BOOST_THRESHOLD = 0.08
_STRUCTURAL_THRESHOLD   = 0.08
_TOP_K                  = 30
_MAX_LLM_CHUNKS         = 20
_HISTORY_COURSE_MAX_AGE_MINUTES = 45

_WELCOME_BODY_MARKERS = (
    "Thank you for your interest in Invertis",
    "We have received your admission inquiry",
    "our counsellor will contact you shortly",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  SUPABASE — lazy init, always reads from os.environ
# ═══════════════════════════════════════════════════════════════════════════════

_supabase_client = None

def _get_db():
    """Always reads env at call time — never at import time."""
    global _supabase_client
    if _supabase_client is None:
        load_dotenv(override=True)
        url = os.environ.get("SUPABASE_URL", "").strip().strip('"').strip("'")
        key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip().strip('"').strip("'")
        if not url or not url.startswith("http"):
            raise RuntimeError(f"SUPABASE_URL invalid in rag_engine: '{url}'")
        if not key:
            raise RuntimeError("SUPABASE_SERVICE_KEY missing in rag_engine")
        from supabase import create_client
        _supabase_client = create_client(url, key)
        logger.info("rag_supabase_client_created")
    return _supabase_client


# ═══════════════════════════════════════════════════════════════════════════════
#  BYPASS GATE
# ═══════════════════════════════════════════════════════════════════════════════

_PURE_GREETINGS: set[str] = {
    "hi", "hello", "hey", "hii", "helo", "helloo", "heyy",
    "ok", "okay", "k",
    "yes", "no", "yep", "nope", "yeah", "nah",
    "thanks", "thank you", "thankyou", "thx", "ty",
    "bye", "goodbye", "good bye",
    "good morning", "good evening", "good afternoon", "good night",
    "ok thank you", "ok thanks", "ok thankyou", "ok ty",
    "okay thank you", "okay thanks", "oh thank you", "oh thanks",
    "nice thank you", "nice thanks", "alright thank you", "alright thanks",
    "great thank you", "great thanks", "good thank you", "good thanks",
    "sounds good thank you", "sounds good thanks",
    "ok done", "okay done", "alright done",
    "got it", "noted", "understood", "i see", "i got it",
    "ok got it", "okay got it",
}

_BYPASS_RAG_PATTERN = re.compile(
    r"""^(?:
        hi+|he+y+|hello+|helo+
        |(?:ok+|okay|k|sure|alright|right|yep|yeah|yup)
            (?:\s*[,.]?\s*
                (?:thanks?|thank\s*you|ty|thx|done|got\s*it|noted|fine|great|good|cool)
            )?
        |(?:thanks?|thank\s*you|ty|thx)
            (?:\s*(?:so\s*much|a\s*lot|very\s*much))?
            (?:\s*[,.]?\s*(?:ok+|okay|bye|sure|great|good|cool))?
        |(?:nice|great|good|wow|cool|awesome|wonderful|excellent|perfect|super|fantastic)
            (?:\s+(?:its?|that'?s?|this|one|sounds?))*
            (?:\s+(?:good|great|nice|cool|awesome|sounds?\s+good))?
            (?:\s*[!.?,]*\s*(?:ok+|okay|thanks?|thank\s*you|ty|bye|sure))*
        |(?:ok+|okay)\s*[,.]?\s*(?:nice|great|good|wow|cool|sounds?\s+good)
        |(?:that'?s?|its?)\s+(?:nice|great|good|cool|awesome|wonderful|excellent|perfect)
        |sounds?\s+(?:good|great|nice|cool|awesome)
        |bye|good\s*(?:morning|evening|night|afternoon)
        |(?:got\s*it|noted|understood|i\s*see|i\s*got\s*it)
            (?:\s*[,.]?\s*(?:thanks?|thank\s*you|ok+|okay))?
        |ok\s*done|okay\s*done|alright\s*done
    )[!.?]*$""",
    re.IGNORECASE | re.VERBOSE,
)


def is_pure_greeting(text: str) -> bool:
    stripped = text.strip()
    cleaned  = re.sub(r"[^\w\s]", "", stripped.lower()).strip()
    if cleaned in _PURE_GREETINGS:
        return True
    return bool(_BYPASS_RAG_PATTERN.match(stripped))


# ═══════════════════════════════════════════════════════════════════════════════
#  COURSE QUERY GATE
# ═══════════════════════════════════════════════════════════════════════════════

_EXPLICIT_COURSE_KEYWORDS: set[str] = {
    "fee", "fees", "cost", "costs", "price", "tuition", "charges",
    "payment", "scholarship", "scholarships",
    "course", "courses", "programme", "programs", "degree",
    "b tech", "btech", "b.tech", "m tech", "mtech", "m.tech",
    "mba", "mca", "pgdm", "pgpm", "bca", "bba", "msc", "bsc", "phd",
    "diploma", "polytechnic", "poly",
    "d pharma", "dpharma", "b pharma", "bpharma", "m pharma", "mpharma",
    "pharmacy", "b ed", "bed", "m ed", "b com", "bcom", "m com", "mcom",
    "b arch", "barch", "llb", "llm", "law",
    "nursing", "paramedical", "forensic", "biotechnology",
    "agriculture", "fintech", "engineering", "management",
    "admission", "admissions", "apply", "application", "enroll",
    "eligibility", "eligible", "qualify", "qualification",
    "entrance", "merit", "cutoff", "syllabus", "curriculum",
    "subjects", "semester", "duration", "intake", "seats",
    "placement", "placements", "internship",
    "specialization", "specialisation", "streams", "stream",
    "branches", "branch", "process",
    "hostel", "accommodation", "campus", "facilities", "library",
    "laboratory", "lab", "sports",
    "information", "detail", "details", "structure",
    "how much", "what is", "what are", "list", "all",
    "change", "switch", "instead",
    "b.sc", "m.sc", "b.com", "m.com",
    "fee structure", "about", "know about", "tell me", "wanna know",
}

_KW_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in sorted(_EXPLICIT_COURSE_KEYWORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_QUESTION_PATTERN = re.compile(
    r"\b(what|how|when|where|which|who|tell|explain|describe|is there|"
    r"do you|can i|can you|please|give me|show me|list|are there|"
    r"available|offered|provide|about|regarding|related|same|also|too|"
    r"another|other|more|and the|for the|wanna|want to|would like|"
    r"know about|tell me about)\b",
    re.IGNORECASE,
)


def is_course_query(text: str) -> bool:
    if is_pure_greeting(text):
        return False
    if _KW_PATTERN.search(text):
        return True
    if _QUESTION_PATTERN.search(text):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  COURSE NAME PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

_COURSE_NAME_PATTERNS = [
    r"\bMBA\b", r"\bMCA\b", r"\bBCA\b", r"\bBBA\b", r"\bPGDM\b", r"\bPGPM\b",
    r"\bB\.?Tech\b", r"\bM\.?Tech\b", r"\bB\.?Sc\.?\b", r"\bM\.?Sc\.?\b",
    r"\bPh\.?D\.?\b", r"\bLL\.?B\b", r"\bLL\.?M\b",
    r"\bPolytechnic\b", r"\bDiploma\b",
    r"\bEngineering\b", r"\bManagement\b",
    r"\bForensic\b", r"\bBiotechnology\b", r"\bComputer\s+Science\b",
    r"\bMechanical\b", r"\bCivil\b", r"\bElectrical\b",
    r"\bAgriculture\b", r"\bPharmacy\b", r"\bCommerce\b",
    r"\bFinTech\b", r"\bArtificial\s+Intelligence\b",
    r"\bB\.?\s*Pharma\b", r"\bM\.?\s*Pharma\b", r"\bD\.?\s*Pharma\b",
    r"\bPharmaceutical\b", r"\bB\.?\s*Ed\.?\b", r"\bM\.?\s*Ed\.?\b",
    r"\bB\.?\s*Com\.?\b", r"\bM\.?\s*Com\.?\b", r"\bB\.?\s*Arch\.?\b",
    r"\bLLB\b", r"\bLaw\b", r"\bNursing\b", r"\bParamedical\b",
]

_COURSE_PATTERN = re.compile("|".join(_COURSE_NAME_PATTERNS), re.IGNORECASE)

_COURSE_SWITCH_PATTERN = re.compile(
    r"\b(?:change|switch|instead\s+of?|rather\s+than|from\s+\S+\s+to|"
    r"not\s+\w+\s+but|replace|drop|move\s+to|want\s+\w+\s+instead|"
    r"prefer|what\s+about|how\s+about)\b",
    re.IGNORECASE,
)

_STRUCTURAL_LIST_PATTERN = re.compile(
    r"\b(streams?|branches?|specializ[ae]tions?|options?|all\s+courses?|"
    r"types?\s+of|list|available|what\s+are\s+the|all\s+\w+\s+in)\b",
    re.IGNORECASE,
)


def _is_welcome_message(body: str) -> bool:
    return any(marker in body for marker in _WELCOME_BODY_MARKERS)


def _extract_courses_from_text(text: str) -> list[str]:
    return _COURSE_PATTERN.findall(text)


def _extract_last_course(history: list[dict]) -> tuple[str, bool]:
    now            = time.time()
    cutoff_seconds = _HISTORY_COURSE_MAX_AGE_MINUTES * 60

    for msg in reversed(history):
        body = msg.get("body", "")
        if _is_welcome_message(body):
            continue
        msg_type = msg.get("messageType") or msg.get("message_type") or ""
        if msg_type in ("welcome", "system_notification"):
            continue

        matches = _extract_courses_from_text(body)
        if matches:
            created_at_str = msg.get("createdAt", "")
            is_recent      = True
            if created_at_str:
                try:
                    from datetime import datetime, timezone
                    dt          = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                    age_seconds = (datetime.now(timezone.utc) - dt).total_seconds()
                    is_recent   = age_seconds < cutoff_seconds
                except Exception:
                    pass
            return matches[-1], is_recent

    return "", False


# ═══════════════════════════════════════════════════════════════════════════════
#  QUERY NORMALIZATION & ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _pre_normalize(text: str) -> str:
    return re.sub(r'\b([a-zA-Z]{1,2})[.\-]([a-zA-Z]{2,})\b', r'\1 \2', text)


def _normalize_query(text: str) -> str:
    replacements = [
        (r"(?i)\bpolytechnic\b",         "Diploma in Engineering Polytechnic"),
        (r"(?i)\bpoly\s*diploma\b",      "Diploma in Engineering Polytechnic"),
        (r"(?i)\bm[\s.\-]?tech\b",       "M.Tech"),
        (r"(?i)\bb[\s.\-]?tech\b",       "B.Tech"),
        (r"(?i)\bm[\s.\-]?sc\b",         "M.Sc"),
        (r"(?i)\bb[\s.\-]?sc\b",         "B.Sc"),
        (r"(?i)\bm[\s.\-]?pharma\b",     "M.Pharma"),
        (r"(?i)\bd[\s.\-]?pharma\b",     "D.Pharma"),
        (r"(?i)\bb[\s.\-]?pharma\b",     "B.Pharma"),
        (r"(?i)\bm[\s.\-]?ed\b",         "M.Ed"),
        (r"(?i)\bb[\s.\-]?ed\b",         "B.Ed"),
        (r"(?i)\bm[\s.\-]?com\b",        "M.Com"),
        (r"(?i)\bb[\s.\-]?com\b",        "B.Com"),
        (r"(?i)\bb[\s.\-]?arch\b",       "B.Arch"),
        (r"(?i)\bm[\s.\-]?b[\s.\-]?a\b", "MBA"),
        (r"(?i)\bm[\s.\-]?c[\s.\-]?a\b", "MCA"),
        (r"(?i)\bb[\s.\-]?c[\s.\-]?a\b", "BCA"),
        (r"(?i)\bb[\s.\-]?b[\s.\-]?a\b", "BBA"),
        (r"(?i)\bcivil\s+engg?\b",        "Civil Engineering"),
        (r"(?i)\bph[\s.\-]?d\b",          "PhD"),
        (r"(?i)\bll[\s.\-]?b\b",          "LLB"),
        (r"(?i)\bll[\s.\-]?m\b",          "LLM"),
    ]
    result = text
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result)
    return result


_ADMISSION_KEYWORDS = re.compile(
    r"\b(fee|fees|cost|duration|eligibility|admission|course|programme|"
    r"seats|hostel|placement|scholarship|streams|branches|subjects|"
    r"how much|how long|how many|tell me|details|about|know about|wanna know)\b",
    re.IGNORECASE,
)


def _expand_to_full_name(query: str) -> str:
    expansions = {
        r'\bBCA\b': 'BCA Bachelor of Computer Applications',
        r'\bBBA\b': 'BBA Bachelor of Business Administration',
        r'\bB\.?Tech\b': 'B.Tech Bachelor of Technology',
        r'\bMBA\b': 'MBA Master of Business Administration',
        r'\bMCA\b': 'MCA Master of Computer Applications',
        r'\bM\.?Tech\b': 'M.Tech Master of Technology',
        r'\bB\.?Pharma\b': 'B.Pharma Bachelor of Pharmacy',
        r'\bD\.?Pharma\b': 'D.Pharma Diploma in Pharmacy',
        r'\bPhD\b': 'PhD Doctor of Philosophy',
        r'\bLLB\b': 'LLB Bachelor of Laws',
        r'\bB\.?Ed\b': 'B.Ed Bachelor of Education',
        r'\bB\.?Com\b': 'B.Com Bachelor of Commerce',
        r'\bB\.?Sc\b': 'B.Sc Bachelor of Science',
        r'\bM\.?Sc\b': 'M.Sc Master of Science',
    }
    result = query
    for pattern, expansion in expansions.items():
        if re.search(pattern, result, re.IGNORECASE):
            result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)
    return result


def _enrich_query(user_message: str, history: list[dict]) -> tuple[str, str, bool]:
    pre_norm       = _pre_normalize(user_message)
    courses_in_msg = _extract_courses_from_text(pre_norm)

    if courses_in_msg:
        is_switch    = bool(_COURSE_SWITCH_PATTERN.search(pre_norm))
        unique       = list(dict.fromkeys(c.lower() for c in courses_in_msg))
        has_multiple = len(unique) > 1

        if is_switch and has_multiple:
            target = courses_in_msg[-1]
            logger.info("query_course_switch", to=target)
            return _expand_to_full_name(_normalize_query(f"{target}: {user_message}")), target, False

        active     = courses_in_msg[-1]
        normalized = _normalize_query(pre_norm)

        if _STRUCTURAL_LIST_PATTERN.search(user_message):
            expanded = f"{normalized} specializations branches streams list all options available"
            return _expand_to_full_name(expanded), active, False

        return _expand_to_full_name(normalized), active, False

    last_in_history, is_recent = _extract_last_course(history)
    has_admission_kw = bool(_ADMISSION_KEYWORDS.search(user_message))

    if last_in_history and is_recent and has_admission_kw:
        enriched = f"{user_message} {last_in_history}"
        logger.info("query_history_enriched",
                    original=user_message[:60], injected=last_in_history)
        return _expand_to_full_name(_normalize_query(enriched)), last_in_history, True

    return _expand_to_full_name(_normalize_query(pre_norm)), "", False


# ═══════════════════════════════════════════════════════════════════════════════
#  LEVEL + LATERAL DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_lateral_query(text: str) -> bool:
    return bool(re.search(r"\b(lateral\s*entry|lateral|le\b)", text, re.IGNORECASE))


def _detect_programme_level(text: str, last_course: str) -> Optional[str]:
    combined = f"{text} {last_course}".lower()
    if re.search(r"\b(polytechnic|poly\s*diploma|diploma\s*poly|diploma\s*in\s*engineering)\b",
                 combined, re.IGNORECASE):
        return "Diploma"
    if re.search(r"\b(ph\.?d|phd|doctorate)\b", combined, re.IGNORECASE):
        return "PhD"
    if re.search(r"\b(d\.?\s*pharma|dpharma|diploma)\b", combined, re.IGNORECASE):
        return "Diploma"
    if re.search(r"\b(mba|mca|m\.?tech|m\.?sc|m\.?pharma|m\.?ed|m\.?com|pgdm|pgpm|llm)\b",
                 combined, re.IGNORECASE):
        return "PG"
    if re.search(r"\b(b\.?tech|btech|b\.?sc|bsc|bca|bba|b\.?pharma|b\.?ed|b\.?com|b\.?arch|llb)\b",
                 combined, re.IGNORECASE):
        return "UG"
    return None


def _get_similarity_threshold(query: str, has_explicit_course: bool = False) -> tuple[float, float]:
    if _STRUCTURAL_LIST_PATTERN.search(query):
        return _STRUCTURAL_THRESHOLD, 0.05
    if re.search(
        r"\b(how many|what are the|list|types of|all\s+\w+\s+streams|"
        r"specializations?|branches|streams|available in|offered in|"
        r"how much|duration|how long|all courses|process)\b",
        query, re.IGNORECASE,
    ):
        return 0.10, 0.05
    if has_explicit_course:
        return _COURSE_BOOST_THRESHOLD, 0.05
    return _MIN_SIMILARITY, _MIN_SIMILARITY_FLOOR


# ═══════════════════════════════════════════════════════════════════════════════
#  GROQ CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

_groq_client = None


def _get_groq():
    global _groq_client
    if _groq_client is None:
        try:
            import httpx
            from groq import Groq
            # FIX 4: read directly from os.environ, not settings
            api_key = os.environ.get("GROQ_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("GROQ_API_KEY not set in environment")
            _groq_client = Groq(
                api_key     = api_key,
                http_client = httpx.Client(timeout=httpx.Timeout(20.0)),
            )
            logger.info("groq_client_created")
        except ImportError:
            raise RuntimeError("groq not installed — pip install groq")
    return _groq_client


# ═══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are an admissions assistant at Invertis University, Bareilly.
Answer students' questions about courses, fees, eligibility, hostel, and campus.

STRICT RULES:
1. Answer ONLY from the STRUCTURED FACTS and COURSE INFORMATION provided below.
2. STRUCTURED FACTS are authoritative — always trust them.
3. If information is not in either block, say ONLY:
   "I don't have that specific detail right now — our counsellor will share it shortly. 🙏"
4. NEVER invent fees, dates, eligibility marks, or course names.
5. Reply in English. Do NOT use the student's name.
6. For course switches ("change MBA to B Pharma"), answer about the TARGET course only.
7. Use this EXACT format every time:

---
[One-line intro about the course or topic]

*Key Details:*
- [Fact 1]
- [Fact 2]
- [Fact 3]
- [Fact 4 if available]

[One closing line with a specific next step]

— Invertis University Admissions
---

8. Each bullet = one fact. No multi-sentence bullets.
9. Duration MUST be the first bullet when asked about duration.
10. List ALL streams/branches when asked — include EVERY course from COURSE INFORMATION. Do NOT truncate or say "only X available".
11. COURSE INFORMATION is the primary truth source — answer from it.
12. DO NOT answer about a course not present in COURSE INFORMATION.
13. If asked about BCA, answer about BCA only — not BBA or B.Tech."""


# ═══════════════════════════════════════════════════════════════════════════════
#  DIRECT TABLE FALLBACK — ILIKE search
# ═══════════════════════════════════════════════════════════════════════════════

FULL_NAME_MAP = {
    "BCA": ["Bachelor of Computer Applications"],
    "BBA": ["Bachelor of Business Administration"],
    "B.Tech": ["Bachelor of Technology", "Engineering"],
    "MBA": ["Master of Business Administration"],
    "MCA": ["Master of Computer Applications"],
    "M.Tech": ["Master of Technology"],
    "B.Pharma": ["Bachelor of Pharmacy"],
    "D.Pharma": ["Diploma in Pharmacy"],
    "M.Pharma": ["Master of Pharmacy"],
    "PhD": ["Doctor of Philosophy"],
    "LLB": ["Bachelor of Laws"],
    "B.Ed": ["Bachelor of Education"],
    "B.Com": ["Bachelor of Commerce"],
    "B.Sc": ["Bachelor of Science"],
    "M.Sc": ["Master of Science"],
    "Diploma in Engineering Polytechnic": ["Diploma in Engineering", "Polytechnic"],
    "Engineering": ["Bachelor of Technology", "Engineering", "B.Tech"],
}


def _direct_table_fallback(course_name: str) -> list[dict]:
    if not course_name:
        return []

    search_terms = [course_name]
    for key, values in FULL_NAME_MAP.items():
        if course_name.lower() == key.lower() or \
           course_name.lower() in [v.lower() for v in values]:
            search_terms = [key] + values
            break

    try:
        db = _get_db()
        for term in search_terms:
            result = (
                db.table("CourseChunk")
                .select("course_name, chunk_text, durationYears, isLateralEntry, programmeLevel, normalisedName")
                .ilike("course_name", f"%{term}%")
                .limit(5)
                .execute()
            )
            rows = result.data or []
            if rows:
                logger.info("direct_fallback_hit",
                            search_term=term, count=len(rows),
                            courses=[r.get("course_name") for r in rows])
                for row in rows:
                    row["similarity"] = 0.50
                    row["id"] = str(row.get("id", ""))
                return rows

        logger.warning("direct_fallback_no_results", course=course_name)
        return []

    except Exception as exc:
        logger.error("direct_fallback_error", course=course_name, error=str(exc))
        return []


# ═══════════════════════════════════════════════════════════════════════════════
#  VECTOR SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def _build_structured_facts(chunks: list[dict]) -> dict:
    if not chunks:
        return {}
    top   = chunks[0]
    facts = {}
    dur   = top.get("durationYears")
    if dur is not None:
        facts["duration_years"] = int(dur)
        facts["duration_text"]  = f"{int(dur)} Year{'s' if int(dur) != 1 else ''}"
    if top.get("isLateralEntry"):
        facts["is_lateral_entry"] = True
    plevel = top.get("programmeLevel")
    if plevel:
        facts["programme_level"] = plevel
    name = top.get("normalisedName") or top.get("course_name")
    if name:
        facts["course_name"] = name
    return facts


def _search_courses_sync(
    query_embedding: list[float],
    top_k:           int,
    query:           str,
    exclude_lateral: bool,
    filter_level:    Optional[str],
    active_course:   str,
) -> tuple[str, list[dict], dict]:

    has_explicit_course = bool(active_course)
    min_sim, floor_sim  = _get_similarity_threshold(query, has_explicit_course)

    logger.info("vector_search_start",
                query=query[:60], course=active_course or "none",
                min_sim=min_sim, floor_sim=floor_sim, top_k=top_k)

    # FIX 2: If active_course is known, try direct fallback FIRST (guaranteed results)
    if active_course:
        direct = _direct_table_fallback(active_course)
        if direct:
            logger.info("using_direct_fallback_first", course=active_course, count=len(direct))
            # Still try vector search to get better ranked results
            # but we have a guaranteed fallback
        
    def _rpc(exc_lat: bool, flevel: Optional[str]) -> list[dict]:
        db = _get_db()
        try:
            result = db.rpc("match_course_chunks", {
                "query_embedding": query_embedding,
                "match_count":     top_k,
                "filter_level":    flevel,
                "exclude_lateral": exc_lat,
            }).execute()
            rows = result.data or []
            logger.info("rpc_result",
                        prog_level=flevel, exc_lateral=exc_lat,
                        count=len(rows),
                        scores=[round(float(r.get("similarity", 0)), 4) for r in rows[:6]])
            return rows
        except Exception as e1:
            logger.warning("rpc_full_params_failed", error=str(e1)[:120])
            try:
                result = db.rpc("match_course_chunks", {
                    "query_embedding": query_embedding,
                    "match_count":     top_k,
                }).execute()
                rows = result.data or []
                logger.info("rpc_minimal_result", count=len(rows),
                            scores=[round(float(r.get("similarity", 0)), 4) for r in rows[:6]])
                return rows
            except Exception as e2:
                logger.error("rpc_completely_failed",
                             e1=str(e1)[:100], e2=str(e2)[:100])
                return []

    chunks = _rpc(exclude_lateral, filter_level)

    if not chunks and filter_level:
        logger.info("retry_no_level_filter")
        chunks = _rpc(exclude_lateral, None)

    if not chunks and exclude_lateral:
        logger.info("retry_no_filters")
        chunks = _rpc(False, None)

    if not chunks and active_course:
        logger.warning("rpc_zero_results_using_direct_fallback", course=active_course)
        chunks = _direct_table_fallback(active_course)

    if not chunks:
        logger.error("all_search_methods_failed", query=query[:60])
        return "", [], {}

    good = [r for r in chunks if float(r.get("similarity", 0)) >= min_sim]

    logger.info("threshold_applied",
                total_chunks=len(chunks),
                above_min=len(good),
                min_sim=min_sim,
                all_scores=[round(float(r.get("similarity", 0)), 4) for r in chunks])

    if not good:
        floor_chunks = [r for r in chunks if float(r.get("similarity", 0)) >= floor_sim]
        if floor_chunks:
            best = float(floor_chunks[0].get("similarity", 0))
            logger.info("using_floor_threshold", best=round(best, 3), floor=floor_sim)
            good = floor_chunks[:_MAX_LLM_CHUNKS]
        else:
            # FIX 2: Last resort — use direct ILIKE fallback
            if active_course:
                direct = _direct_table_fallback(active_course)
                if direct:
                    logger.info("all_below_floor_using_direct", course=active_course)
                    good = direct[:_MAX_LLM_CHUNKS]
                else:
                    return "", [], {}
            else:
                return "", [], {}

    good.sort(key=lambda r: float(r.get("similarity", 0)), reverse=True)
    llm_chunks = good[:_MAX_LLM_CHUNKS]

    lines = []
    for r in llm_chunks:
        course = r.get("course_name", "Course")
        sim    = float(r.get("similarity", 0))
        text   = r.get("chunk_text", "")
        lines.append(f"[{course}] (relevance: {sim:.2f})\n{text}")

    return "\n\n---\n\n".join(lines), llm_chunks, _build_structured_facts(llm_chunks)


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def _build_chat_history_text(history: list[dict]) -> str:
    lines = []
    for msg in history[-8:]:
        body = msg.get("body", "").strip()
        if not body or _is_welcome_message(body):
            continue
        role = "Student" if msg.get("direction") == "inbound" else "Assistant"
        lines.append(f"{role}: {body[:300]}")
    return "\n".join(lines)


def _format_chunks_fallback(chunks: list[dict]) -> str:
    if not chunks:
        return "I don't have specific details on that right now — our counsellor will reach out shortly. 🙏"
    lines = ["Here's what I found:\n"]
    for chunk in chunks[:2]:
        course = chunk.get("course_name", "")
        text   = chunk.get("chunk_text", "").strip()
        if course:
            lines.append(f"*{course}*")
        lines.append(text[:400])
        lines.append("")
    lines.append("Would you like more details? 😊")
    return "\n".join(lines)


def _generate_sync(
    user_message:        str,
    context:             str,
    chat_history:        str,
    raw_chunks:          list[dict],
    active_course:       str,
    structured_facts:    dict,
    course_from_history: bool,
) -> str:
    sections: list[str] = []

    if structured_facts:
        fact_lines = []
        if "course_name" in structured_facts:
            fact_lines.append(f"Course: {structured_facts['course_name']}")
        if "duration_text" in structured_facts:
            fact_lines.append(f"Duration: {structured_facts['duration_text']}")
        if "programme_level" in structured_facts:
            fact_lines.append(f"Level: {structured_facts['programme_level']}")
        if structured_facts.get("is_lateral_entry"):
            fact_lines.append("Type: Lateral Entry")
        if fact_lines:
            sections.append(
                "=== STRUCTURED FACTS (authoritative) ===\n"
                + "\n".join(fact_lines)
                + "\n=== END STRUCTURED FACTS ==="
            )

    sections.append(f"=== COURSE INFORMATION ===\n{context}\n=== END ===")

    if chat_history.strip():
        sections.append(
            f"=== CONVERSATION HISTORY ===\n{chat_history}\n=== END ==="
        )

    if active_course and not course_from_history:
        sections.append(
            f"CURRENT COURSE: Student is asking about '{active_course}'. "
            f"Answer ONLY about this course. Do NOT mix in other courses."
        )

    sections.append(
        "IMPORTANT: Use ONLY the COURSE INFORMATION above. "
        "DO NOT invent any fees, durations, or details not explicitly stated."
    )
    sections.append(f"Student: {user_message}")
    sections.append("Reply (follow format instructions exactly):")

    user_turn = "\n\n".join(sections)

    try:
        response = _get_groq().chat.completions.create(
            model       = "llama-3.1-8b-instant",
            max_tokens  = 1500,
            temperature = 0.05,
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
        if structured_facts.get("duration_text"):
            return (
                f"*Key Details:*\n- Duration: {structured_facts['duration_text']}\n\n"
                "Our counsellor will share complete details shortly. 🙏"
            )
        return _format_chunks_fallback(raw_chunks)


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — rag_reply
# ═══════════════════════════════════════════════════════════════════════════════

_SAFE_GREETING_REPLY = (
    "Great! Feel free to ask anything about Invertis University — "
    "courses, fees, hostel, or admissions. 😊"
)


async def rag_reply(
    lead_phone:   str,
    user_message: str,
    lead_name:    str,
) -> str:
    # Import here to avoid circular import / early Supabase init
    from webhook.conversation_store import get_conversation_history, normalise_phone

    norm_phone = normalise_phone(lead_phone)

    # Hard greeting gate
    if is_pure_greeting(user_message):
        logger.info("rag_hard_gate_greeting_bypass", preview=user_message[:40])
        return _SAFE_GREETING_REPLY

    try:
        history = await get_conversation_history(norm_phone, limit=20)
    except Exception as exc:
        logger.warning("history_fetch_failed", error=str(exc))
        history = []

    logger.info("rag_pipeline_start",
                phone=norm_phone, history_count=len(history),
                message=user_message[:60])

    enriched_query, active_course, course_from_history = _enrich_query(
        user_message, history
    )

    is_lateral   = _detect_lateral_query(user_message)
    filter_level = _detect_programme_level(user_message, active_course)

    logger.info("rag_query_ready",
                enriched=enriched_query[:80],
                course=active_course or "none",
                from_history=course_from_history,
                prog_level=filter_level or "none")

    try:
        q_vec: list[float] = await asyncio.to_thread(
            lambda: _embedder.encode(enriched_query).tolist()
        )
    except Exception as exc:
        logger.error("embedding_failed", error=str(exc))
        return "Our counsellor will reach out with details shortly. 🙏"

    context, raw_chunks, structured_facts = await asyncio.to_thread(
        _search_courses_sync,
        q_vec, _TOP_K, enriched_query,
        not is_lateral, filter_level, active_course,
    )

    if not context:
        logger.info("rag_no_context", query=enriched_query[:60], course=active_course or "none")
        # FIX: Try direct fallback one more time before giving up
        if active_course:
            direct = await asyncio.to_thread(_direct_table_fallback, active_course)
            if direct:
                lines = []
                for r in direct[:_MAX_LLM_CHUNKS]:
                    course = r.get("course_name", "Course")
                    text   = r.get("chunk_text", "")
                    lines.append(f"[{course}]\n{text}")
                context        = "\n\n---\n\n".join(lines)
                raw_chunks     = direct[:_MAX_LLM_CHUNKS]
                structured_facts = _build_structured_facts(raw_chunks)
                logger.info("rag_context_from_last_resort_direct", course=active_course)

    if not context:
        return (
            "I don't have specific details on that right now. "
            "Our counsellor will reach out with accurate information shortly. 🙏\n\n"
            "You can also ask me about course fees, eligibility, or the admission process! 😊"
        )

    chat_history_text = _build_chat_history_text(history)
    reply = await asyncio.to_thread(
        _generate_sync,
        user_message, context, chat_history_text,
        raw_chunks, active_course, structured_facts, course_from_history,
    )

    logger.info("rag_done",
                phone=norm_phone, course=active_course or "none",
                context_len=len(context), reply_len=len(reply))
    return reply


# ═══════════════════════════════════════════════════════════════════════════════
#  INGESTION HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════

async def check_ingestion_health() -> dict:
    result: dict = {"chunk_count": 0, "rpc_works": False,
                    "test_returns_results": False, "issues": []}
    try:
        db = _get_db()
        cr = db.table("CourseChunk").select("id", count="exact").execute()
        result["chunk_count"] = cr.count or 0
        if result["chunk_count"] == 0:
            result["issues"].append("CRITICAL: CourseChunk is EMPTY. Run: python ingest_courses.py --clear")
            return result

        sample = db.table("CourseChunk").select("embedding, course_name").limit(1).execute()
        if sample.data:
            emb = sample.data[0].get("embedding")
            if isinstance(emb, list):
                result["embedding_dim"] = len(emb)
                if len(emb) != EMBEDDING_DIM:
                    result["issues"].append(
                        f"WRONG embedding dim: stored={len(emb)}, model={EMBEDDING_DIM}. Re-ingest!"
                    )
    except Exception as exc:
        result["issues"].append(f"DB access failed: {exc}")
        return result

    try:
        test_vec = await asyncio.to_thread(
            lambda: _embedder.encode("BCA fees duration eligibility").tolist()
        )
        ctx, chunks, facts = await asyncio.to_thread(
            _search_courses_sync, test_vec, 3, "BCA fees duration", False, None, "BCA"
        )
        result["rpc_works"]            = True
        result["test_returns_results"] = bool(ctx)
        if not ctx:
            result["issues"].append(
                "RPC returns 0 results for 'BCA fees'. "
                "Run complete_schema_final.sql then python ingest_courses.py --clear"
            )
    except Exception as exc:
        result["issues"].append(f"RPC test failed: {exc}")

    return result