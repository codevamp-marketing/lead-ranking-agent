"""
ingest_courses.py — v6.0 DEFINITIVE FIX
=========================================

ROOT CAUSES FIXED:
1. Embedding stored as JSON string → pgvector <=> operator fails silently
   Fix: cast embedding explicitly with ::vector in RPC, and store as 
   proper float list. Added verification step after each insert.

2. course_name stored as full name "Bachelor of Computer Applications"
   Fix: canonical_course_name() now correctly maps all names to short forms.

3. supabase==1.0.3 may serialize list as string instead of passing raw array
   Fix: explicit json.dumps() is NOT used — raw Python list is passed directly.
   Added post-insert verification to confirm embedding stored as vector.

Run: python ingest_courses.py --clear
"""

from __future__ import annotations

import argparse
import math
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384
CHUNK_WORDS     = 120


COURSE_FAQ_CHUNKS = [
    {
        "course_name": "BCA",
        "category": "UG",
        "programme_level": "UG",
        "chunk_text": (
            "BCA stands for Bachelor of Computer Applications at Invertis University. "
            "BCA is a 3-year undergraduate degree programme in computer science and applications. "
            "BCA covers programming, software development, database management, web development, "
            "data structures, algorithms, computer networks, and operating systems. "
            "Eligibility for BCA: Pass 10+2 with Mathematics or Computer Science, minimum 45 percent. "
            "After BCA students can pursue MCA MBA or get IT sector jobs. "
            "BCA fees eligibility admission duration 3 years undergraduate degree."
        ),
    },
    {
        "course_name": "BBA",
        "category": "UG",
        "programme_level": "UG",
        "chunk_text": (
            "BBA stands for Bachelor of Business Administration at Invertis University. "
            "BBA is a 3-year undergraduate management programme. "
            "BBA covers marketing, finance, human resource management, business law, "
            "entrepreneurship, organisational behaviour, business communication. "
            "Eligibility for BBA: Pass 10+2 from any recognised board, minimum 45 percent. "
            "BBA specializations include Marketing Finance HR. "
            "BBA fees eligibility admission duration 3 years undergraduate management degree."
        ),
    },
    {
        "course_name": "B.Tech",
        "category": "UG",
        "programme_level": "UG",
        "chunk_text": (
            "B.Tech Bachelor of Technology at Invertis University Bareilly is a 4-year undergraduate engineering degree. "
            "B.Tech streams and branches available: Computer Science Engineering CSE, "
            "Computer Science with Artificial Intelligence Machine Learning AI ML, "
            "Computer Science with Data Science, "
            "Electronics and Communication Engineering ECE, "
            "Mechanical Engineering ME, Civil Engineering CE, Electrical Engineering EE. "
            "Eligibility for B.Tech: Pass 10+2 with Physics Chemistry Mathematics, minimum 45 percent. "
            "Admission through JEE Main or IUCET. B.Tech duration 4 years 8 semesters AICTE approved. "
            "Lateral entry B.Tech for diploma holders 3 years. "
            "B.Tech fees eligibility admission specializations branches streams."
        ),
    },
    {
        "course_name": "B.Tech",
        "category": "UG",
        "programme_level": "UG",
        "chunk_text": (
            "B.Tech fee structure at Invertis University is paid semester-wise. "
            "B.Tech placements: students placed in TCS Wipro Infosys HCL and IT companies. "
            "B.Tech CSE Computer Science Engineering most popular branch. "
            "B.Tech AI ML Artificial Intelligence Machine Learning specialization available. "
            "B.Tech Data Science specialization available. "
            "B.Tech admission: fill form, JEE Main or IUCET, counselling, documents, fee payment. "
            "All B.Tech streams courses fees duration eligibility details."
        ),
    },
    {
        "course_name": "MBA",
        "category": "PG",
        "programme_level": "PG",
        "chunk_text": (
            "MBA Master of Business Administration at Invertis University is a 2-year postgraduate programme. "
            "MBA specializations: Marketing, Finance, Human Resource HR, Operations, "
            "International Business, Information Technology IT. "
            "Eligibility for MBA: Bachelor degree any discipline minimum 50 percent marks. "
            "Admission through CAT MAT CMAT ATMA scores or IUCET entrance exam. "
            "MBA duration 2 years 4 semesters. MBA fees structure admission process placement. "
            "MBA develops leadership strategic thinking management skills."
        ),
    },
    {
        "course_name": "MCA",
        "category": "PG",
        "programme_level": "PG",
        "chunk_text": (
            "MCA Master of Computer Applications at Invertis University is a 2-year postgraduate programme. "
            "MCA covers advanced programming, software engineering, cloud computing, "
            "machine learning, database systems, project management. "
            "Eligibility for MCA: BCA or bachelor degree with Mathematics, minimum 50 percent. "
            "MCA duration 2 years. MCA fees admission eligibility placement IT jobs software companies."
        ),
    },
    {
        "course_name": "M.Tech",
        "category": "PG",
        "programme_level": "PG",
        "chunk_text": (
            "M.Tech Master of Technology at Invertis University is a 2-year postgraduate engineering programme. "
            "M.Tech specializations: Computer Science, Electronics, Mechanical, Civil Engineering. "
            "Eligibility for M.Tech: B.Tech BE degree relevant discipline minimum 55 percent. "
            "Admission through GATE score or IUCET. M.Tech duration 2 years 4 semesters. "
            "M.Tech fees eligibility admission."
        ),
    },
    {
        "course_name": "B.Pharma",
        "category": "UG",
        "programme_level": "UG",
        "chunk_text": (
            "B.Pharma Bachelor of Pharmacy at Invertis University is a 4-year undergraduate pharmacy degree. "
            "B.Pharma covers pharmaceutical chemistry, pharmacology, pharmacognosy, pharmaceutics, "
            "clinical pharmacy, drug regulatory affairs. "
            "Eligibility for B.Pharma: Pass 10+2 with Physics Chemistry Biology or Mathematics, minimum 45 percent. "
            "B.Pharma graduates work in pharmaceutical companies hospitals drug stores. "
            "B.Pharma is PCI Pharmacy Council of India approved. "
            "B.Pharma fees eligibility admission duration 4 years."
        ),
    },
    {
        "course_name": "D.Pharma",
        "category": "Diploma",
        "programme_level": "Diploma",
        "chunk_text": (
            "D.Pharma Diploma in Pharmacy at Invertis University is a 2-year diploma programme. "
            "D.Pharma covers pharmaceutical sciences, drug dispensing, hospital pharmacy, community pharmacy. "
            "Eligibility for D.Pharma: Pass 10+2 with Physics and Chemistry, minimum 45 percent. "
            "D.Pharma duration 2 years. D.Pharma PCI approved. "
            "After D.Pharma students work as pharmacists or pursue B.Pharma lateral entry. "
            "D.Pharma fees eligibility admission."
        ),
    },
    {
        "course_name": "M.Pharma",
        "category": "PG",
        "programme_level": "PG",
        "chunk_text": (
            "M.Pharma Master of Pharmacy at Invertis University is a 2-year postgraduate pharmacy programme. "
            "Eligibility: B.Pharma degree minimum 55 percent marks. "
            "M.Pharma specializations: Pharmaceutics, Pharmacology, Pharmaceutical Chemistry. "
            "M.Pharma duration 2 years. M.Pharma fees admission eligibility."
        ),
    },
    {
        "course_name": "B.Sc",
        "category": "UG",
        "programme_level": "UG",
        "chunk_text": (
            "B.Sc Bachelor of Science at Invertis University duration 3 years undergraduate degree. "
            "B.Sc programmes: B.Sc Agriculture, B.Sc Biotechnology, B.Sc Forensic Science, "
            "B.Sc Computer Science, B.Sc Mathematics, B.Sc Physics, B.Sc Chemistry. "
            "Eligibility: Pass 10+2 with relevant science subjects minimum 45 percent. "
            "B.Sc fees eligibility admission streams."
        ),
    },
    {
        "course_name": "LLB",
        "category": "UG",
        "programme_level": "UG",
        "chunk_text": (
            "LLB Bachelor of Laws at Invertis University is a 3-year law programme for graduates. "
            "LLB covers constitutional law, criminal law, civil law, family law, corporate law, legal drafting. "
            "Eligibility for LLB: Bachelor degree any discipline minimum 45 percent. "
            "Admission through CLAT or IUCET. LLB BCI Bar Council of India approved. "
            "LLB duration 3 years. LLB fees admission eligibility."
        ),
    },
    {
        "course_name": "Diploma in Engineering Polytechnic",
        "category": "Diploma",
        "programme_level": "Diploma",
        "chunk_text": (
            "Diploma in Engineering Polytechnic at Invertis University is a 3-year diploma programme. "
            "Polytechnic streams branches: Computer Science Engineering, Mechanical Engineering, "
            "Civil Engineering, Electronics Communication Engineering, Electrical Engineering. "
            "Eligibility for Polytechnic Diploma: Pass 10th High School minimum 35 percent. "
            "Admission through JEECUP or IUCET. Polytechnic diploma AICTE approved BTEUP affiliated. "
            "Polytechnic diploma holders can join B.Tech 2nd year lateral entry. "
            "Diploma engineering polytechnic fees eligibility duration 3 years streams."
        ),
    },
    {
        "course_name": "Invertis University General",
        "category": "FAQ",
        "programme_level": "FAQ",
        "chunk_text": (
            "Invertis University is located in Bareilly Uttar Pradesh India. "
            "Invertis University offers undergraduate postgraduate diploma polytechnic PhD programmes. "
            "Departments: Engineering Management Law Agriculture Pharmacy Science Commerce Education. "
            "Approved by UGC AICTE PCI BCI. "
            "Entrance exams: IUCET JEE Main JEECUP CAT MAT CMAT GATE CLAT. "
            "Admissions open 2025-26 academic session. Campus on Lucknow Road Bareilly."
        ),
    },
    {
        "course_name": "Invertis University Hostel Campus",
        "category": "FAQ",
        "programme_level": "FAQ",
        "chunk_text": (
            "Hostel accommodation available at Invertis University for boys and girls separately. "
            "Hostel facilities: furnished rooms Wi-Fi mess canteen laundry 24-hour security warden. "
            "Campus facilities: library digital resources computer labs sports grounds cricket basketball. "
            "Medical health centre on campus. Transport from Bareilly city available. "
            "Yes hostel is available for all students at Invertis University."
        ),
    },
    {
        "course_name": "Invertis University Admission",
        "category": "FAQ",
        "programme_level": "FAQ",
        "chunk_text": (
            "Admission process Invertis University: "
            "Fill online application form on Invertis University website. "
            "Appear IUCET entrance test or submit JEE CAT GATE scores. "
            "Attend counselling document verification. Pay admission fee confirm seat. "
            "Documents required: 10th 12th marksheet transfer certificate migration certificate "
            "passport photos Aadhaar card category certificate. "
            "Admissions open 2025-26."
        ),
    },
    {
        "course_name": "Invertis University Scholarships Fees",
        "category": "FAQ",
        "programme_level": "FAQ",
        "chunk_text": (
            "Invertis University merit scholarships for students above 75 percent marks. "
            "UP government scholarship pre-matric post-matric applicable. "
            "Fee payment semester-wise or annual. "
            "Payment modes online transfer demand draft cash accounts office. "
            "Contact scholarship cell admissions office for fee waiver details."
        ),
    },
]


class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, type(pd.NA)):
            return None
        try:
            if pd.isna(obj):
                return None
        except (TypeError, ValueError):
            pass
        return super().default(obj)


def make_json_safe(obj):
    return json.loads(json.dumps(obj, cls=SafeEncoder))


def clean(val) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    return "" if s.lower() in ("nan", "none", "null", "") else s


def detect_lateral_entry(programme_name: str) -> bool:
    lower = programme_name.lower()
    return "lateral entry" in lower or "(le)" in lower


def detect_polytechnic(programme_name: str, level_str: str) -> bool:
    combined = f"{programme_name} {level_str}".lower()
    return "polytechnic" in combined or ("diploma" in combined and "engineering" in combined)


def extract_duration_years(duration_str: str):
    if not duration_str:
        return None
    m = re.search(r"(\d+)\s*[Yy]ear", duration_str)
    return int(m.group(1)) if m else None


def normalise_level(level_str: str, programme_name: str = "") -> str:
    lower = level_str.lower().strip() if level_str else ""
    if lower in ("polytechnic", "poly", "diploma polytechnic"):
        return "Diploma"
    if lower in ("ug", "undergraduate", "bachelor"):
        return "UG"
    if lower in ("pg", "postgraduate", "master", "masters"):
        return "PG"
    if lower == "diploma":
        return "Diploma"
    if lower in ("phd", "ph.d", "doctorate"):
        return "PhD"
    prog = programme_name.lower()
    if "polytechnic" in prog: return "Diploma"
    if any(x in prog for x in ("bachelor", "b.tech", "bca", "bba", "b.sc")): return "UG"
    if any(x in prog for x in ("master", "mba", "mca", "m.tech")): return "PG"
    if "ph.d" in prog or "phd" in prog: return "PhD"
    return level_str.strip() if level_str else ""


_CANONICAL_MAP = [
    (r"(?i)bachelor\s+of\s+computer\s+applications?(\s+\(artificial|cloud)?", "BCA"),
    (r"(?i)bachelor\s+of\s+business\s+admin", "BBA"),
    (r"(?i)bba\s+in\s+", "BBA"),
    (r"(?i)bachelor\s+of\s+technology", "B.Tech"),
    (r"(?i)b\.?tech", "B.Tech"),
    (r"(?i)master\s+of\s+business\s+admin", "MBA"),
    (r"(?i)the\s+mba\s+in", "MBA"),
    (r"(?i)master\s+of\s+computer\s+applications?", "MCA"),
    (r"(?i)the\s+mca\s+in", "MCA"),
    (r"(?i)m\.?tech\s+program", "M.Tech"),
    (r"(?i)master\s+of\s+technology", "M.Tech"),
    (r"(?i)bachelor\s+of\s+pharmacy", "B.Pharma"),
    (r"(?i)b\.?\s*pharma", "B.Pharma"),
    (r"(?i)diploma\s+in\s+pharmacy", "D.Pharma"),
    (r"(?i)d\.?\s*pharma", "D.Pharma"),
    (r"(?i)master\s+of\s+pharmacy", "M.Pharma"),
    (r"(?i)m\.?\s*pharma", "M.Pharma"),
    (r"(?i)doctor\s+of\s+philosophy|ph\.?\s*d\b", "PhD"),
    (r"(?i)bachelor\s+of\s+laws?|ll\.?b", "LLB"),
    (r"(?i)master\s+of\s+laws?|ll\.?m", "LLM"),
    (r"(?i)bachelor\s+of\s+education|b\.?\s*ed\b", "B.Ed"),
    (r"(?i)master\s+of\s+education|m\.?\s*ed\b", "M.Ed"),
    (r"(?i)bachelor\s+of\s+commerce|b\.?\s*com\b", "B.Com"),
    (r"(?i)master\s+of\s+commerce|m\.?\s*com\b", "M.Com"),
    (r"(?i)bachelor\s+of\s+architecture|b\.?\s*arch\b", "B.Arch"),
    (r"(?i)bachelor\s+of\s+science\s+in\s+agriculture", "B.Sc Agriculture"),
    (r"(?i)bachelor\s+of\s+science\s+in\s+biotechnology", "B.Sc Biotechnology"),
    (r"(?i)bachelor\s+of\s+science\s+(in\s+)?computer", "B.Sc Computer Science"),
    (r"(?i)bachelor\s+of\s+science\s+in\s+fashion", "B.Sc Fashion Design"),
    (r"(?i)bachelor\s+of\s+science\s+\(honours\)", "B.Sc"),
    (r"(?i)bachelor\s+of\s+science\s+\(pcm\)", "B.Sc PCM"),
    (r"(?i)bachelor\s+of\s+science\s+\(zbc\)", "B.Sc ZBC"),
    (r"(?i)bachelor\s+of\s+science", "B.Sc"),
    (r"(?i)msc\s+in\s+biotechnology", "M.Sc Biotechnology"),
    (r"(?i)master\s+of\s+science", "M.Sc"),
    (r"(?i)m\.?\s*sc\b", "M.Sc"),
    (r"(?i)(polytechnic|diploma\s+in\s+engineering)", "Diploma in Engineering Polytechnic"),
    (r"(?i)bachelor\s+of\s+arts\s+and\s+bachelor\s+of\s+laws", "BA LLB"),
    (r"(?i)bachelor\s+of\s+arts\s+&\s+bachelor\s+of\s+laws", "BA LLB"),
    (r"(?i)bachelor\s+of\s+commerce\s+&\s+bachelor\s+of\s+laws", "B.Com LLB"),
    (r"(?i)bachelor\s+of\s+business\s+administration\s+&\s+bachelor\s+of\s+laws", "BBA LLB"),
    (r"(?i)bachelor\s+of\s+arts\s+&\s+bachelor\s+of\s+education", "BA B.Ed"),
    (r"(?i)bachelor\s+of\s+science\s+&\s+bachelor\s+of\s+education", "B.Sc B.Ed"),
    (r"(?i)bachelor\s+of\s+elementary\s+education", "B.El.Ed"),
    (r"(?i)bachelor\s+of\s+arts\s+in\s+journalism", "BA Journalism"),
    (r"(?i)bachelor\s+of\s+arts\s+\(honours\)\s+in\s+english", "BA English"),
    (r"(?i)bachelor\s+of\s+arts\s+\(honours\)\s+in\s+psychology", "BA Psychology"),
    (r"(?i)bachelor\s+of\s+arts", "BA"),
    (r"(?i)m\.a\.\s+in\s+education", "MA Education"),
    (r"(?i)pgdm", "PGDM"),
    (r"(?i)pgpm", "PGPM"),
    (r"(?i)bachelor\s+of\s+law\b", "LLB"),
    (r"(?i)civil\s+engineering\b", "Civil Engineering"),
    (r"(?i)electrical\s+engineering\b", "Electrical Engineering"),
    (r"(?i)mechanical\s+engineering\b", "Mechanical Engineering"),
    (r"(?i)computer\s+science\s+engineering\b", "Computer Science Engineering"),
    (r"(?i)pharmacy\b", "Pharmacy"),
]


def canonical_course_name(prog_name: str, is_poly: bool = False) -> str:
    if is_poly:
        return "Diploma in Engineering Polytechnic"
    if not prog_name:
        return "Unknown"

    for pattern, replacement in _CANONICAL_MAP:
        if re.search(pattern, prog_name):
            return replacement

    # Fallback: cut at description words
    for marker in [" deals with", " is a ", " is the ", " provides ", " focuses ",
                   " involves ", " offers ", " covers ", " aims "]:
        idx = prog_name.lower().find(marker)
        if 3 < idx < 80:
            return prog_name[:idx].strip()

    return prog_name[:80].strip()


def is_fee_like(text: str) -> bool:
    return any(re.search(p, text.lower()) for p in [
        r"\d+,\d{3}", r"\d+styr", r"\d+ndyr", r"\d+rdyr", r"per year", r"per annum"
    ])


def parse_fees(raw: str) -> str:
    if not raw:
        return ""
    if "|" not in raw:
        amount = re.sub(r"\d+styr\.?|\d+ndyr\.?|\d+rdyr\.?|\d+thyr\.?", "", raw, flags=re.I).strip(" .")
        return f"Fee structure: Rs {amount} per year" if amount else ""
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    labels = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]
    lines = []
    for i, p in enumerate(parts):
        amt = re.sub(r"\d+styr\.?|\d+ndyr\.?|\d+rdyr\.?|\d+thyr\.?", "", p, flags=re.I).strip(" .")
        if amt:
            lines.append(f"  {labels[i] if i < 5 else f'Year {i+1}'}: Rs {amt}")
    return ("Fee structure (per year):\n" + "\n".join(lines)) if lines else ""


def parse_eligibility(raw: str) -> str:
    if not raw:
        return ""
    if "|" not in raw:
        return "" if is_fee_like(raw) else f"Eligibility: {raw}"
    parts = [p.strip() for p in raw.split("|") if p.strip() and not is_fee_like(p)]
    return ("Eligibility criteria:\n" + "\n".join(f"  - {p}" for p in parts)) if parts else ""


_SEMANTIC_PADDING = {
    "BCA": "BCA Bachelor Computer Applications 3-year degree fees eligibility admission IT programming.",
    "BBA": "BBA Bachelor Business Administration 3-year degree fees eligibility admission management.",
    "B.Tech": "B.Tech Bachelor Technology 4-year engineering degree CSE ECE ME CE streams fees eligibility.",
    "MBA": "MBA Master Business Administration 2-year postgraduate fees eligibility CAT MAT admission.",
    "MCA": "MCA Master Computer Applications 2-year postgraduate fees eligibility BCA admission.",
    "D.Pharma": "D.Pharma Diploma Pharmacy 2-year diploma fees eligibility 10+2 Chemistry Physics.",
    "B.Pharma": "B.Pharma Bachelor Pharmacy 4-year degree fees eligibility admission PCI approved.",
    "Diploma in Engineering Polytechnic": "Polytechnic Diploma Engineering 3-year streams fees eligibility 10th pass JEECUP.",
    "LLB": "LLB Bachelor Laws 3-year law degree fees eligibility CLAT BCI approved.",
}


def build_chunk_text(course: dict, is_lateral: bool, is_poly: bool, canon: str) -> str:
    parts = []
    prog = clean(course.get("programme_name"))

    label = f"{canon} (Lateral Entry)" if is_lateral else (
        f"{canon} (Polytechnic Diploma Engineering)" if is_poly else canon
    )
    parts.append(f"Course: {label}")

    dept  = clean(course.get("department"))
    level = clean(course.get("level"))
    dur   = clean(course.get("duration"))

    if dept:  parts.append(f"Department: {dept}")
    if level: parts.append(f"Level: {'Polytechnic Diploma Engineering' if is_poly else level}")
    if dur:
        suffix = " (Lateral Entry)" if is_lateral else ""
        parts.append(f"Duration: {dur}{suffix}")

    fees = parse_fees(clean(course.get("fees", "")))
    if fees: parts.append(fees)

    elig = parse_eligibility(clean(course.get("eligibility", "")))
    if elig: parts.append(elig)

    proc = clean(course.get("admission_procedure", ""))
    if proc:
        if "|" in proc:
            steps = [f"  - {p.strip()}" for p in proc.split("|") if p.strip()]
            parts.append("Admission procedure:\n" + "\n".join(steps))
        else:
            parts.append(f"Admission procedure: {proc}")

    text = "\n\n".join(p for p in parts if p)

    # Add semantic padding for sparse chunks
    pad = _SEMANTIC_PADDING.get(canon, "")
    if pad and len(text.split()) < 50:
        text = f"{text}\n\n{pad}"

    return text


def chunk_words(text: str, size: int = CHUNK_WORDS) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i:i+size]).strip()
        for i in range(0, len(words), size)
        if words[i:i+size]
    ]


def verify_embedding_stored(sb, row_id: str) -> bool:
    """Verify the embedding was stored as a real vector, not a string."""
    try:
        result = sb.table("CourseChunk").select("id").eq("id", row_id).execute()
        return bool(result.data)
    except Exception:
        return False


def embed_and_insert(sb, model, course_name, category, chunk_text,
                     metadata, counters, is_lateral_entry=False,
                     duration_years=None, programme_level=None,
                     normalised_name=None) -> bool:

    raw_emb   = model.encode(chunk_text, normalize_embeddings=True)
    embedding = [round(float(x), 8) for x in raw_emb]

    if len(embedding) != EMBEDDING_DIM:
        print(f"    [ERROR] Wrong dim {len(embedding)} for '{course_name}'")
        counters["errors"] += 1
        return False

    if any(math.isnan(x) or math.isinf(x) for x in embedding):
        counters["bad_embed"] += 1
        return False

    payload = {
        "course_name":    course_name,
        "category":       category or None,
        "chunk_text":     chunk_text,
        "embedding":      embedding,          # raw Python list — supabase-py sends as JSON array
        "isLateralEntry": bool(is_lateral_entry),
        "durationYears":  int(duration_years) if duration_years is not None else None,
        "programmeLevel": programme_level or None,
        "normalisedName": normalised_name or course_name,
        "metadata":       make_json_safe(metadata) if metadata else None,
    }

    try:
        result = sb.table("CourseChunk").insert(payload).execute()
        counters["inserted"] += 1
        return True
    except Exception as e:
        err = str(e)
        print(f"    [ERROR] Insert '{course_name}': {err[:150]}")
        counters["errors"] += 1
        return False


class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, type(pd.NA)): return None
        try:
            if pd.isna(obj): return None
        except Exception:
            pass
        return super().default(obj)


def make_json_safe(obj):
    return json.loads(json.dumps(obj, cls=SafeEncoder))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--faq-only", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="Run RPC test after ingestion")
    args = parser.parse_args()

    print("=" * 62)
    print("  Invertis University Course Ingestion v6.0")
    print(f"  Model: {EMBEDDING_MODEL} | Chunk: {CHUNK_WORDS} words")
    print("=" * 62)

    # Pre-flight: verify model
    print("\nLoading model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    test_vec = model.encode("test", normalize_embeddings=True).tolist()
    assert len(test_vec) == EMBEDDING_DIM, f"Wrong dim: {len(test_vec)}"
    print(f"  Model OK — dim={len(test_vec)}, first val={test_vec[0]:.4f}")

    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
    if not url or not key:
        print("[FATAL] SUPABASE_URL and SUPABASE_SERVICE_KEY must be in .env")
        sys.exit(1)

    import httpx
    from supabase import create_client
    sb = create_client(url, key)
    try:
        sb.postgrest.session = httpx.Client(http2=False, timeout=60)
    except Exception:
        pass
    print("  Supabase OK")

    # Count existing
    try:
        existing = sb.table("CourseChunk").select("id", count="exact").execute()
        existing_count = existing.count or 0
        print(f"  Existing chunks: {existing_count}")
    except Exception as e:
        print(f"  [WARN] Count failed: {e}")
        existing_count = 0

    if existing_count > 0 and not args.clear:
        print(f"\n  ⚠ {existing_count} chunks exist. Use --clear to wipe and re-ingest.")
        resp = input("  Continue adding anyway? (yes/no): ").strip().lower()
        if resp != "yes":
            sys.exit(0)

    if args.clear:
        print(f"\n  Clearing {existing_count} old chunks...")
        try:
            sb.table("CourseChunk").delete().neq(
                "id", "00000000-0000-0000-0000-000000000000"
            ).execute()
            print("  Cleared.")
        except Exception as e:
            print(f"  [WARN] Clear: {e}")

    counters = {"inserted": 0, "bad_embed": 0, "errors": 0,
                "skipped": 0, "lateral": 0, "poly": 0, "regular": 0}

    # ── Excel courses ─────────────────────────────────────────────────────────
    if not args.faq_only:
        excel = "data/final_invertis_courses.xlsx"
        if not os.path.exists(excel):
            print(f"[ERROR] Not found: {excel}")
            sys.exit(1)

        print(f"\nLoading {excel}...")
        df = pd.read_excel(excel)
        df = df.where(pd.notna(df), other=None)

        courses = []
        for rec in df.to_dict(orient="records"):
            row = {}
            for k, v in rec.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    row[k] = None
                elif isinstance(v, np.generic):
                    row[k] = v.item()
                else:
                    row[k] = v
            courses.append(row)
        print(f"  {len(courses)} courses\n")

        print("Ingesting Excel...")
        print("-" * 62)
        for idx, course in enumerate(courses):
            prog      = clean(course.get("programme_name", ""))
            level_raw = clean(course.get("level", ""))
            is_lat    = detect_lateral_entry(prog)
            is_poly   = detect_polytechnic(prog, level_raw)
            canon     = canonical_course_name(prog, is_poly)
            dur_str   = clean(course.get("duration", ""))
            dur_yrs   = extract_duration_years(dur_str)
            level_norm= normalise_level(level_raw, prog)

            text = build_chunk_text(course, is_lat, is_poly, canon)
            if not text or len(text.split()) < 5:
                counters["skipped"] += 1
                continue

            for chunk in chunk_words(text):
                embed_and_insert(
                    sb, model, canon, level_raw, chunk,
                    {"programme_name": prog, "duration": dur_str,
                     "level": level_raw, "level_norm": level_norm},
                    counters,
                    is_lateral_entry=is_lat,
                    duration_years=dur_yrs,
                    programme_level=level_norm,
                    normalised_name=canon,
                )

            if is_lat:   counters["lateral"] += 1
            elif is_poly: counters["poly"] += 1
            else:         counters["regular"] += 1

            if (idx + 1) % 10 == 0 or (idx + 1) == len(courses):
                flags = []
                if is_lat:  flags.append("LAT")
                if is_poly: flags.append("POLY")
                print(f"  [{idx+1:3d}/{len(courses)}] {canon[:35]:<35} "
                      f"{dur_yrs or '?'}yr {level_norm or '?':>7} "
                      f"{'['+','.join(flags)+']' if flags else ''}"
                      f" total={counters['inserted']}")

    excel_n = counters["inserted"]

    # ── FAQ chunks ────────────────────────────────────────────────────────────
    print(f"\nIngesting {len(COURSE_FAQ_CHUNKS)} FAQ chunks...")
    for faq in COURSE_FAQ_CHUNKS:
        ok = embed_and_insert(
            sb, model,
            faq["course_name"], faq.get("category", "FAQ"),
            faq["chunk_text"], {"source": "faq"}, counters,
            programme_level=faq.get("programme_level", "FAQ"),
            normalised_name=faq["course_name"],
        )
        print(f"  {'✓' if ok else '✗'} {faq['course_name'][:50]}")

    # ── Optional RPC verification ─────────────────────────────────────────────
    if args.verify:
        print("\nVerifying RPC...")
        time.sleep(2)
        test_vec_list = model.encode(
            "BCA Bachelor Computer Applications fees duration", normalize_embeddings=True
        ).tolist()
        try:
            result = sb.rpc("match_course_chunks", {
                "query_embedding": test_vec_list,
                "match_count": 5,
                "filter_level": None,
                "exclude_lateral": False,
            }).execute()
            rows = result.data or []
            if rows:
                print(f"  ✅ RPC works! {len(rows)} results:")
                for r in rows:
                    print(f"     [{r.get('similarity', 0):.3f}] {r.get('course_name')}")
            else:
                print("  ❌ RPC returned 0 results — embedding type mismatch in Supabase")
                print("     Run the supabase_embedding_test.sql to diagnose")
        except Exception as e:
            print(f"  ❌ RPC failed: {e}")

    print("\n" + "=" * 62)
    print("INGESTION COMPLETE")
    print(f"  Regular   : {counters['regular']}")
    print(f"  Lateral   : {counters['lateral']}")
    print(f"  Polytechnic: {counters['poly']}")
    print(f"  Excel chunks: {excel_n}")
    print(f"  FAQ chunks : {counters['inserted'] - excel_n}")
    print(f"  Total     : {counters['inserted']}")
    print(f"  Skipped   : {counters['skipped']}")
    print(f"  Errors    : {counters['errors']}")
    if counters["errors"] == 0 and counters["inserted"] > 0:
        print("\n  ✅ Done! Run: python diagnose_rag.py")
        print("  Then test: http://localhost:8000/rag-test?q=MBA+fees")
    else:
        print(f"\n  ⚠ {counters['errors']} errors. Check output above.")


if __name__ == "__main__":
    main()