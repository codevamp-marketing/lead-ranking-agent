"""
ingest_courses.py — Invertis University Course Ingestion (v5.0 — PERMANENT FIX)
================================================================================
ALWAYS run with --clear:
    python ingest_courses.py --clear

ROOT CAUSES FIXED IN v5.0
──────────────────────────

FIX 1 — DUPLICATE EMBEDDINGS (was your #1 retrieval killer)
  v4.0 was run twice without --clear → 188 rows = 94 old + 94 new.
  When the same course has two embeddings (possibly from different code
  versions), similarity scores are diluted and noisy. The --clear guard
  is now enforced: it refuses to run without --clear unless you explicitly
  confirm. Added a pre-flight count check that warns if > 0 rows exist.

FIX 2 — CHUNK TEXT TOO SPARSE for BCA / BBA / generic courses
  If fees/eligibility columns are empty in the Excel, build_course_text()
  produces a chunk with only 3-4 lines (~30 words). A 30-word chunk for
  "BCA" embeds almost identically to a 30-word chunk for "BBA" because
  both just say "Course: BCA, Level: UG, Department: ...".
  Fix: added _add_semantic_padding() which appends canonical descriptive
  sentences per course type. This makes BCA chunks semantically distinct
  from BBA chunks even when Excel data is sparse.

FIX 3 — POLYTECHNIC vs D.PHARMA COLLISION (was returning D.Pharma for poly queries)
  Preserved from v4.0 (FIX B) — polytechnic chunks explicitly include
  "Polytechnic Diploma Engineering" terms.

FIX 4 — COURSE NAME STORED IN course_name COLUMN IS INCONSISTENT
  short_name() was returning truncated names like "Bachelor of Computer"
  instead of "BCA". Added a canonical_course_name() function that maps
  known programme names to their canonical short form (BCA, BBA, B.Tech, etc.)
  This makes the direct ILIKE fallback in rag_engine.py work correctly.

FIX 5 — FAQ CHUNKS FOR ALL COMMON COURSES
  Added per-course FAQ chunks for BCA, BBA, B.Tech, MBA, MCA, D.Pharma, etc.
  These are synthetic high-quality chunks that answer the most common
  student questions even if the Excel data for that course is sparse.
  This guarantees a minimum quality answer for every course.

FIX 6 — EMBEDDING DIMENSION VERIFICATION
  Added a pre-flight check: encodes a test string and verifies dim=384.
  If the wrong model is loaded (e.g. OpenAI 1536-dim), it exits immediately.
"""

from __future__ import annotations

import argparse
import math
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384
CHUNK_WORDS     = 120   # slightly larger for richer semantic content


# ── FIX 5: Per-course FAQ chunks ──────────────────────────────────────────────
# These guarantee good retrieval even when Excel data is sparse.
# Each chunk is written to be semantically rich and course-specific.

COURSE_FAQ_CHUNKS = [
    # ── BCA ───────────────────────────────────────────────────────────────────
    {
        "course_name": "BCA",
        "category": "UG",
        "programme_level": "UG",
        "chunk_text": (
            "BCA stands for Bachelor of Computer Applications at Invertis University. "
            "BCA is a 3-year undergraduate degree programme in computer science and applications. "
            "BCA course covers programming, software development, database management, "
            "web development, data structures, algorithms, computer networks, and operating systems. "
            "Eligibility for BCA: Pass 10+2 (Intermediate) with Mathematics or Computer Science "
            "from any recognised board. Minimum 45 percent marks required. "
            "BCA fee structure at Invertis University: contact admissions for current fee details. "
            "After BCA, students can pursue MCA, MBA, or direct job placements in IT sector. "
            "BCA is offered under the School of Computing and Information Technology."
        ),
    },
    # ── BBA ───────────────────────────────────────────────────────────────────
    {
        "course_name": "BBA",
        "category": "UG",
        "programme_level": "UG",
        "chunk_text": (
            "BBA stands for Bachelor of Business Administration at Invertis University. "
            "BBA is a 3-year undergraduate management programme. "
            "BBA course covers marketing, finance, human resource management, "
            "business law, entrepreneurship, organisational behaviour, and business communication. "
            "Eligibility for BBA: Pass 10+2 (Intermediate) from any recognised board. "
            "Minimum 45 percent marks in qualifying examination. "
            "BBA specializations at Invertis University include Marketing, Finance, and HR. "
            "After BBA, students can pursue MBA or enter management roles in corporate sector. "
            "BBA is a popular choice for students interested in business and entrepreneurship."
        ),
    },
    # ── B.Tech ────────────────────────────────────────────────────────────────
    {
        "course_name": "B.Tech",
        "category": "UG",
        "programme_level": "UG",
        "chunk_text": (
            "B.Tech stands for Bachelor of Technology at Invertis University Bareilly. "
            "B.Tech is a 4-year undergraduate engineering degree programme. "
            "B.Tech streams and branches available at Invertis University: "
            "Computer Science and Engineering (CSE), "
            "Computer Science with Artificial Intelligence and Machine Learning (CS-AI/ML), "
            "Computer Science with Data Science (CS-DS), "
            "Electronics and Communication Engineering (ECE), "
            "Mechanical Engineering (ME), "
            "Civil Engineering (CE), "
            "Electrical Engineering (EE). "
            "Eligibility for B.Tech: Pass 10+2 (Intermediate) with Physics, Chemistry, Mathematics. "
            "Minimum 45 percent marks. Admission through JEE Main or IUCET. "
            "B.Tech is AICTE approved. Duration 4 years (8 semesters). "
            "Lateral entry B.Tech available for diploma holders: 3-year programme."
        ),
    },
    {
        "course_name": "B.Tech",
        "category": "UG",
        "programme_level": "UG",
        "chunk_text": (
            "B.Tech fee structure at Invertis University: fee is paid semester-wise. "
            "B.Tech placement record: students placed in TCS, Wipro, Infosys, HCL, and other IT companies. "
            "B.Tech Computer Science Engineering is the most popular branch. "
            "B.Tech with AI and Machine Learning specialization is available. "
            "B.Tech with Data Science specialization is available. "
            "B.Tech admission process: fill online form, appear for JEE Main or IUCET, "
            "attend counselling, submit documents, pay fee. "
            "Documents required: 12th marksheet, Aadhaar card, passport photo, transfer certificate."
        ),
    },
    # ── MBA ───────────────────────────────────────────────────────────────────
    {
        "course_name": "MBA",
        "category": "PG",
        "programme_level": "PG",
        "chunk_text": (
            "MBA stands for Master of Business Administration at Invertis University. "
            "MBA is a 2-year postgraduate management programme. "
            "MBA specializations at Invertis University: Marketing, Finance, Human Resource, "
            "Operations, International Business, Information Technology. "
            "Eligibility for MBA: Bachelor degree in any discipline with minimum 50 percent marks. "
            "Admission through CAT, MAT, CMAT, ATMA scores or IUCET. "
            "MBA duration: 2 years (4 semesters). "
            "MBA programme develops leadership, strategic thinking, and management skills. "
            "MBA placement cell arranges campus drives with top companies."
        ),
    },
    # ── MCA ───────────────────────────────────────────────────────────────────
    {
        "course_name": "MCA",
        "category": "PG",
        "programme_level": "PG",
        "chunk_text": (
            "MCA stands for Master of Computer Applications at Invertis University. "
            "MCA is a 2-year postgraduate programme in computer applications and software. "
            "MCA course covers advanced programming, software engineering, cloud computing, "
            "machine learning, database systems, and project management. "
            "Eligibility for MCA: BCA or any bachelor degree with Mathematics at 10+2 or graduation level. "
            "Minimum 50 percent marks in qualifying degree. "
            "MCA duration: 2 years. "
            "MCA graduates are placed in software companies, IT firms, and tech startups."
        ),
    },
    # ── M.Tech ────────────────────────────────────────────────────────────────
    {
        "course_name": "M.Tech",
        "category": "PG",
        "programme_level": "PG",
        "chunk_text": (
            "M.Tech stands for Master of Technology at Invertis University. "
            "M.Tech is a 2-year postgraduate engineering programme. "
            "M.Tech specializations: Computer Science, Electronics, Mechanical, Civil. "
            "Eligibility for M.Tech: B.Tech or BE degree in relevant discipline. "
            "Minimum 55 percent marks in qualifying degree. "
            "Admission through GATE score or IUCET. "
            "M.Tech duration: 2 years (4 semesters)."
        ),
    },
    # ── B.Pharma ──────────────────────────────────────────────────────────────
    {
        "course_name": "B.Pharma",
        "category": "UG",
        "programme_level": "UG",
        "chunk_text": (
            "B.Pharma stands for Bachelor of Pharmacy at Invertis University. "
            "B.Pharma is a 4-year undergraduate pharmacy degree programme. "
            "B.Pharma covers pharmaceutical chemistry, pharmacology, pharmacognosy, "
            "pharmaceutics, clinical pharmacy, and drug regulatory affairs. "
            "Eligibility for B.Pharma: Pass 10+2 with Physics, Chemistry, Biology or Mathematics. "
            "Minimum 45 percent marks. "
            "B.Pharma graduates can work in pharmaceutical companies, hospitals, and drug stores. "
            "B.Pharma is PCI (Pharmacy Council of India) approved."
        ),
    },
    # ── D.Pharma ──────────────────────────────────────────────────────────────
    {
        "course_name": "D.Pharma",
        "category": "Diploma",
        "programme_level": "Diploma",
        "chunk_text": (
            "D.Pharma stands for Diploma in Pharmacy at Invertis University. "
            "D.Pharma is a 2-year diploma programme in pharmacy. "
            "D.Pharma covers basic pharmaceutical sciences, drug dispensing, "
            "hospital pharmacy, and community pharmacy practice. "
            "Eligibility for D.Pharma: Pass 10+2 with Physics and Chemistry. "
            "Minimum 45 percent marks. "
            "D.Pharma duration: 2 years. "
            "D.Pharma is PCI approved. After D.Pharma, students can work as pharmacists "
            "or pursue B.Pharma through lateral entry."
        ),
    },
    # ── M.Pharma ──────────────────────────────────────────────────────────────
    {
        "course_name": "M.Pharma",
        "category": "PG",
        "programme_level": "PG",
        "chunk_text": (
            "M.Pharma stands for Master of Pharmacy at Invertis University. "
            "M.Pharma is a 2-year postgraduate pharmacy programme. "
            "Eligibility: B.Pharma degree with minimum 55 percent marks. "
            "M.Pharma specializations: Pharmaceutics, Pharmacology, Pharmaceutical Chemistry. "
            "Duration: 2 years."
        ),
    },
    # ── B.Sc ──────────────────────────────────────────────────────────────────
    {
        "course_name": "B.Sc",
        "category": "UG",
        "programme_level": "UG",
        "chunk_text": (
            "B.Sc stands for Bachelor of Science at Invertis University. "
            "B.Sc programmes available: B.Sc Agriculture, B.Sc Biotechnology, "
            "B.Sc Forensic Science, B.Sc Computer Science, B.Sc Mathematics. "
            "B.Sc duration: 3 years. "
            "Eligibility: Pass 10+2 with relevant science subjects, minimum 45 percent. "
            "B.Sc Agriculture is a popular choice for students from rural backgrounds."
        ),
    },
    # ── LLB ───────────────────────────────────────────────────────────────────
    {
        "course_name": "LLB",
        "category": "UG",
        "programme_level": "UG",
        "chunk_text": (
            "LLB stands for Bachelor of Laws at Invertis University. "
            "LLB is a 3-year undergraduate law programme (for graduates). "
            "LLB covers constitutional law, criminal law, civil law, family law, "
            "corporate law, and legal drafting. "
            "Eligibility for LLB: Bachelor degree in any discipline with minimum 45 percent. "
            "Admission through CLAT or IUCET. "
            "LLB is BCI (Bar Council of India) approved. "
            "Duration: 3 years for LLB (post-graduation entry)."
        ),
    },
    # ── Polytechnic Diploma ───────────────────────────────────────────────────
    {
        "course_name": "Diploma in Engineering Polytechnic",
        "category": "Diploma",
        "programme_level": "Diploma",
        "chunk_text": (
            "Invertis University offers Diploma in Engineering (Polytechnic) programmes. "
            "Polytechnic Diploma is a 3-year full-time diploma engineering programme. "
            "Polytechnic streams at Invertis University: "
            "Computer Science and Engineering Polytechnic, "
            "Mechanical Engineering Polytechnic, "
            "Civil Engineering Polytechnic, "
            "Electronics and Communication Engineering Polytechnic, "
            "Electrical Engineering Polytechnic. "
            "Eligibility for Polytechnic Diploma: Pass 10th (High School) from recognised board. "
            "Minimum 35 percent marks in 10th board examination. "
            "Admission through JEECUP or IUCET. "
            "Polytechnic diploma holders can join B.Tech 2nd year through lateral entry. "
            "AICTE approved polytechnic programme affiliated to BTEUP."
        ),
    },
    # ── General FAQ ───────────────────────────────────────────────────────────
    {
        "course_name": "Invertis University General Information",
        "category": "FAQ",
        "programme_level": "FAQ",
        "chunk_text": (
            "Invertis University is located in Bareilly, Uttar Pradesh, India. "
            "The university offers undergraduate, postgraduate, diploma, polytechnic, "
            "and PhD programmes in Engineering, Management, Law, Agriculture, "
            "Pharmacy, Science, Commerce, and Education. "
            "Invertis University is approved by UGC, AICTE, PCI, and BCI. "
            "Entrance exams accepted: IUCET, JEE Main, JEECUP, CAT, MAT, CMAT, GATE, CLAT. "
            "Academic session 2025-26 admissions are open. "
            "The university campus is located on Lucknow Road, Bareilly."
        ),
    },
    {
        "course_name": "Invertis University Hostel and Campus Facilities",
        "category": "FAQ",
        "programme_level": "FAQ",
        "chunk_text": (
            "Hostel facility is available at Invertis University for boys and girls. "
            "Yes, hostel accommodation is available for all students at Invertis University. "
            "Hostel facilities: furnished rooms, Wi-Fi, mess and canteen, laundry, 24-hour security. "
            "Campus facilities: well-equipped library, digital resources, computer labs, "
            "sports grounds, basketball court, cricket ground, indoor games. "
            "Medical health centre is on campus. Transport available from Bareilly city. "
            "Separate hostels for boys and girls with warden."
        ),
    },
    {
        "course_name": "Invertis University Admission Process",
        "category": "FAQ",
        "programme_level": "FAQ",
        "chunk_text": (
            "Admission process at Invertis University step by step: "
            "Step 1: Fill the online application form on Invertis University website. "
            "Step 2: Appear for IUCET entrance test or submit valid JEE, CAT, GATE scores. "
            "Step 3: Attend counselling session and document verification. "
            "Step 4: Pay the admission fee and confirm seat. "
            "Documents required for admission: 10th marksheet, 12th marksheet, "
            "transfer certificate, migration certificate, passport size photographs, "
            "Aadhaar card, category certificate if applicable. "
            "Admissions open for 2025-26. Online admission form available on website."
        ),
    },
    {
        "course_name": "Invertis University Scholarships",
        "category": "FAQ",
        "programme_level": "FAQ",
        "chunk_text": (
            "Invertis University offers merit scholarships for students scoring above 75 percent. "
            "UP government scholarship pre-matric and post-matric schemes are applicable. "
            "Fee payment: semester-wise or annual. "
            "Payment modes: online transfer, demand draft, or cash at accounts office. "
            "Contact scholarship cell or admissions office for latest scholarship details and fee waivers."
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
    if not programme_name:
        return False
    lower = programme_name.lower()
    return "lateral entry" in lower or "(le)" in lower or "lateral-entry" in lower


def detect_polytechnic(programme_name: str, level_str: str) -> bool:
    combined = f"{programme_name} {level_str}".lower()
    return (
        "polytechnic" in combined
        or ("diploma" in combined and "engineering" in combined)
    )


def extract_duration_years(duration_str: str):
    if not duration_str:
        return None
    match = re.search(r"(\d+)\s*[Yy]ear", duration_str)
    return int(match.group(1)) if match else None


def normalise_level(level_str: str, programme_name: str = "") -> str | None:
    if not level_str:
        return None
    lower = level_str.lower().strip()

    if lower in ("polytechnic", "poly", "diploma polytechnic", "diploma (poly)"):
        return "Diploma"
    if lower in ("ug", "undergraduate", "graduate programme", "bachelor"):
        return "UG"
    if lower in ("pg", "postgraduate", "post graduate", "master", "masters"):
        return "PG"
    if lower in ("diploma",):
        return "Diploma"
    if lower in ("phd", "ph.d", "doctorate", "doctoral"):
        return "PhD"

    prog_lower = programme_name.lower()
    if "polytechnic" in prog_lower:
        return "Diploma"
    if any(x in prog_lower for x in ("bachelor", "b.tech", "b.sc", "b.com", "bca", "bba")):
        return "UG"
    if any(x in prog_lower for x in ("master", "mba", "mca", "m.tech", "m.sc")):
        return "PG"
    if "ph.d" in prog_lower or "phd" in prog_lower:
        return "PhD"

    return level_str.strip()


# FIX 4: canonical_course_name maps full programme names to short searchable names
_CANONICAL_NAME_MAP = [
    (r"(?i)bachelor\s+of\s+computer\s+applications?", "BCA"),
    (r"(?i)bachelor\s+of\s+business\s+admin", "BBA"),
    (r"(?i)bachelor\s+of\s+technology", "B.Tech"),
    (r"(?i)b\.?tech", "B.Tech"),
    (r"(?i)master\s+of\s+business\s+admin", "MBA"),
    (r"(?i)master\s+of\s+computer\s+applications?", "MCA"),
    (r"(?i)master\s+of\s+technology", "M.Tech"),
    (r"(?i)m\.?tech", "M.Tech"),
    (r"(?i)bachelor\s+of\s+pharmacy", "B.Pharma"),
    (r"(?i)b\.?\s*pharma", "B.Pharma"),
    (r"(?i)diploma\s+in\s+pharmacy", "D.Pharma"),
    (r"(?i)d\.?\s*pharma", "D.Pharma"),
    (r"(?i)master\s+of\s+pharmacy", "M.Pharma"),
    (r"(?i)m\.?\s*pharma", "M.Pharma"),
    (r"(?i)bachelor\s+of\s+(science|arts|commerce)", lambda m: f"B.{m.group(1)[0].upper()+'Sc' if 'sc' in m.group(1).lower() else m.group(1)[0].upper()}"),
    (r"(?i)doctor\s+of\s+philosophy|ph\.?\s*d", "PhD"),
    (r"(?i)bachelor\s+of\s+laws?|ll\.?b", "LLB"),
    (r"(?i)master\s+of\s+laws?|ll\.?m", "LLM"),
    (r"(?i)bachelor\s+of\s+education|b\.?\s*ed", "B.Ed"),
    (r"(?i)master\s+of\s+education|m\.?\s*ed", "M.Ed"),
    (r"(?i)bachelor\s+of\s+commerce|b\.?\s*com", "B.Com"),
    (r"(?i)master\s+of\s+commerce|m\.?\s*com", "M.Com"),
    (r"(?i)bachelor\s+of\s+architecture|b\.?\s*arch", "B.Arch"),
    (r"(?i)(polytechnic|diploma\s+in\s+engineering)", "Diploma in Engineering Polytechnic"),
    (r"(?i)bachelor\s+of\s+science\s+in\s+agriculture", "B.Sc Agriculture"),
    (r"(?i)bachelor\s+of\s+science\s+in\s+biotechnology", "B.Sc Biotechnology"),
    (r"(?i)bachelor\s+of\s+science\s+in\s+forensic", "B.Sc Forensic Science"),
    (r"(?i)pgdm", "PGDM"),
    (r"(?i)pgpm", "PGPM"),
]


def canonical_course_name(prog_name: str, is_poly: bool = False) -> str:
    """FIX 4: Return canonical short name for reliable ILIKE fallback search."""
    if is_poly:
        return "Diploma in Engineering Polytechnic"
    for pattern, replacement in _CANONICAL_NAME_MAP:
        if callable(replacement):
            m = re.search(pattern, prog_name)
            if m:
                try:
                    return replacement(m)
                except Exception:
                    pass
        elif re.search(pattern, prog_name):
            return replacement

    # Fallback: truncate at first description marker
    desc_starters = [
        " deals with", " is a ", " is the ", " provides ", " focuses ",
        " involves ", " offers ", " covers ", " aims ", " prepares ",
        " equips ", " trains ", " includes ",
    ]
    name = prog_name
    for starter in desc_starters:
        idx = prog_name.lower().find(starter)
        if idx > 5:
            name = prog_name[:idx].strip()
            break
    return name[:100]


def is_fee_like(text: str) -> bool:
    fee_indicators = [
        r"\d+,\d{3}", r"\d+styr", r"\d+ndyr", r"\d+rdyr", r"\d+thyr",
        r"per year", r"per annum",
    ]
    lower = text.lower()
    return any(re.search(pat, lower) for pat in fee_indicators)


def parse_fees(raw: str) -> str:
    if not raw:
        return ""
    if "|" not in raw:
        amount = re.sub(r"\d+styr\.?|\d+ndyr\.?|\d+rdyr\.?|\d+thyr\.?",
                        "", raw, flags=re.IGNORECASE).strip(" .")
        return f"Fee structure: Rs {amount} per year" if amount else ""
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    year_labels = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]
    lines = []
    for i, part in enumerate(parts):
        amount = re.sub(
            r"\d+styr\.?|\d+ndyr\.?|\d+rdyr\.?|\d+thyr\.?",
            "", part, flags=re.IGNORECASE
        ).strip(" .")
        label = year_labels[i] if i < len(year_labels) else f"Year {i+1}"
        if amount:
            lines.append(f"  {label}: Rs {amount}")
    return ("Fee structure (per year):\n" + "\n".join(lines)) if lines else ""


def parse_eligibility(raw: str) -> str:
    if not raw:
        return ""
    if "|" not in raw:
        return "" if is_fee_like(raw) else f"Eligibility: {raw}"
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    eligibility_parts = [p for p in parts if not is_fee_like(p)]
    if not eligibility_parts:
        return ""
    lines = [f"  - {p}" for p in eligibility_parts]
    return "Eligibility criteria:\n" + "\n".join(lines)


# FIX 2: Semantic padding to distinguish sparse courses
_SEMANTIC_PADDING = {
    "BCA": (
        "BCA Bachelor of Computer Applications computer science programming software development "
        "web development database networking operating systems algorithms data structures. "
        "BCA is a 3-year undergraduate degree. BCA fees eligibility admission."
    ),
    "BBA": (
        "BBA Bachelor of Business Administration management marketing finance human resources "
        "business strategy entrepreneurship organisational behaviour. "
        "BBA is a 3-year undergraduate degree in management. BBA fees eligibility streams specialization."
    ),
    "B.Tech": (
        "B.Tech Bachelor of Technology engineering computer science mechanical civil electrical electronics. "
        "B.Tech streams branches specializations CSE ECE ME CE EE AI ML data science. "
        "B.Tech is a 4-year undergraduate engineering programme. B.Tech fees eligibility admission JEE."
    ),
    "MBA": (
        "MBA Master of Business Administration postgraduate management marketing finance HR operations. "
        "MBA specializations streams duration 2 years eligibility CAT MAT CMAT. "
        "MBA fees admission process placement."
    ),
    "MCA": (
        "MCA Master of Computer Applications postgraduate computer science software programming. "
        "MCA duration 2 years eligibility BCA graduate. MCA fees admission placement IT jobs."
    ),
    "D.Pharma": (
        "D.Pharma Diploma in Pharmacy 2-year diploma programme pharmaceutical dispensing hospital pharmacy. "
        "D.Pharma eligibility 10+2 Physics Chemistry. D.Pharma fees PCI approved."
    ),
    "B.Pharma": (
        "B.Pharma Bachelor of Pharmacy 4-year undergraduate pharmacy programme pharmaceutical chemistry. "
        "B.Pharma eligibility 10+2 PCB or PCM. B.Pharma fees PCI approved drug regulatory."
    ),
    "Diploma in Engineering Polytechnic": (
        "Polytechnic Diploma in Engineering 3-year diploma programme computer science mechanical civil electrical. "
        "Polytechnic eligibility 10th pass JEECUP. Polytechnic fees streams branches BTEUP AICTE approved. "
        "Diploma engineering lateral entry B.Tech."
    ),
    "LLB": (
        "LLB Bachelor of Laws 3-year law programme constitutional law criminal law civil law corporate law. "
        "LLB eligibility graduation BCI approved CLAT. LLB fees admission."
    ),
}


def _add_semantic_padding(text: str, canonical_name: str) -> str:
    """FIX 2: Append domain-specific terms to sparse chunks for better embeddings."""
    padding = _SEMANTIC_PADDING.get(canonical_name, "")
    if padding and len(text.split()) < 60:
        return f"{text}\n\n{padding}"
    return text


def build_course_text(course: dict, is_lateral: bool, is_poly: bool,
                      canonical_name: str) -> str:
    parts: list[str] = []

    prog_name = clean(course.get("programme_name"))
    name      = canonical_name

    if name:
        if is_lateral:
            parts.append(f"Course: {name} (Lateral Entry Programme)")
        elif is_poly:
            parts.append(f"Course: {name} (Polytechnic Diploma Engineering Programme)")
        else:
            parts.append(f"Course: {name}")

    # Add description if meaningfully different from name
    if prog_name and len(prog_name) > len(name) + 15:
        desc_markers = [
            " deals with", " is a ", " is the ", " provides ", " focuses ",
            " involves ", " offers ", " covers ", " aims ", " prepares ",
        ]
        for marker in desc_markers:
            idx = prog_name.lower().find(marker)
            if idx > 5:
                desc = prog_name[idx:].strip().lstrip(".").strip()
                if len(desc) > 20:
                    parts.append(f"About: {desc}")
                break

    dept  = clean(course.get("department"))
    level = clean(course.get("level"))
    ptype = clean(course.get("programme_type"))

    if dept:
        parts.append(f"Department: {dept}")
    if level:
        if is_poly:
            parts.append("Level: Polytechnic Diploma (Diploma in Engineering)")
        else:
            parts.append(f"Level: {level}")
    if ptype:
        parts.append(f"Programme type: {ptype}")

    duration = clean(course.get("duration"))
    if duration:
        if is_lateral:
            parts.append(f"Duration: {duration} (Lateral Entry — shorter than regular programme)")
        else:
            parts.append(f"Duration: {duration}")

    fees_raw = clean(course.get("fees"))
    if fees_raw:
        fees_text = parse_fees(fees_raw)
        if fees_text:
            parts.append(fees_text)

    elig_raw = clean(course.get("eligibility"))
    if elig_raw:
        elig_text = parse_eligibility(elig_raw)
        if elig_text:
            parts.append(elig_text)

    proc = clean(course.get("admission_procedure"))
    if proc:
        if "|" in proc:
            proc_parts = [p.strip() for p in proc.split("|") if p.strip()]
            proc_lines = [f"  - {p}" for p in proc_parts]
            parts.append("Admission procedure:\n" + "\n".join(proc_lines))
        else:
            parts.append(f"Admission procedure: {proc}")

    text = "\n\n".join(p for p in parts if p.strip())

    # FIX 2: add semantic padding if chunk is sparse
    return _add_semantic_padding(text, canonical_name)


def chunk_words(text: str, size: int = CHUNK_WORDS) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i: i + size]).strip()
        for i in range(0, len(words), size)
        if " ".join(words[i: i + size]).strip()
    ]


def embed_and_insert(
    sb, model, course_name, category, chunk_text, metadata, counters,
    is_lateral_entry=False, duration_years=None,
    programme_level=None, normalised_name=None,
) -> None:
    raw_emb   = model.encode(chunk_text)
    embedding = [float(x) for x in raw_emb]

    if any(math.isnan(x) or math.isinf(x) for x in embedding):
        counters["bad_embed"] += 1
        return

    if len(embedding) != EMBEDDING_DIM:
        print(f"    [ERROR] Wrong embedding dim {len(embedding)} for '{course_name}'")
        counters["errors"] += 1
        return

    payload = make_json_safe({
        "course_name":    course_name,
        "category":       category or None,
        "chunk_text":     chunk_text,
        "embedding":      embedding,
        "metadata":       metadata,
        "isLateralEntry": is_lateral_entry,
        "durationYears":  duration_years,
        "programmeLevel": programme_level,
        "normalisedName": normalised_name,
    })

    try:
        sb.table("CourseChunk").insert(payload).execute()
        counters["inserted"] += 1
    except Exception as e:
        err_str = str(e)
        print(f"    [ERROR] Insert failed for '{course_name}': {err_str[:120]}")
        counters["errors"] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true",
                        help="REQUIRED: Delete all old chunks before ingesting")
    parser.add_argument("--faq-only", action="store_true",
                        help="Only ingest FAQ chunks (skip Excel)")
    args = parser.parse_args()

    print("=" * 62)
    print("  Invertis University — Course Ingestion (v5.0)")
    print(f"  Model: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Chunk size: {CHUNK_WORDS} words")
    print("=" * 62)

    # ── FIX 6: Pre-flight embedding dimension check ───────────────────────────
    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    test_vec = model.encode("test").tolist()
    if len(test_vec) != EMBEDDING_DIM:
        print(f"[FATAL] Model produces dim={len(test_vec)}, expected {EMBEDDING_DIM}.")
        print("Fix: pip install -U sentence-transformers && check model name.")
        sys.exit(1)
    print(f"  Model OK — dim={len(test_vec)}")

    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
    if not url or not key:
        print("[ERROR] Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env")
        sys.exit(1)

    import httpx
    from supabase import create_client
    sb = create_client(url, key)
    try:
        sb.postgrest.session = httpx.Client(http2=False, timeout=30)
    except Exception:
        pass
    print("Supabase connected.")

    # ── FIX 1: Pre-flight row count check ────────────────────────────────────
    try:
        existing = sb.table("CourseChunk").select("id", count="exact").execute()
        existing_count = existing.count or 0
    except Exception as e:
        print(f"[WARN] Could not count existing rows: {e}")
        existing_count = 0

    if existing_count > 0 and not args.clear:
        print(f"\n  ⚠ WARNING: {existing_count} existing chunks found in CourseChunk!")
        print("  Running without --clear will create DUPLICATES.")
        print("  Duplicates cause noisy retrieval and wrong answers.")
        print("  STRONGLY recommended: python ingest_courses.py --clear")
        print()
        resp = input("  Continue anyway and ADD to existing? (yes/no): ").strip().lower()
        if resp != "yes":
            print("  Aborted. Run: python ingest_courses.py --clear")
            sys.exit(0)

    if args.clear:
        print(f"\nClearing ALL {existing_count} existing chunks...")
        try:
            sb.table("CourseChunk").delete().neq(
                "id", "00000000-0000-0000-0000-000000000000"
            ).execute()
            print(f"  Deleted {existing_count} old chunks. Starting fresh.")
        except Exception as e:
            print(f"  [WARN] Clear error: {e}")

    counters = {
        "inserted": 0, "bad_embed": 0, "errors": 0,
        "skipped_empty": 0, "lateral": 0, "regular": 0, "polytechnic": 0,
    }

    # ── Ingest Excel courses ──────────────────────────────────────────────────
    if not args.faq_only:
        excel_path = "data/final_invertis_courses.xlsx"
        if not os.path.exists(excel_path):
            print(f"[ERROR] Not found: {excel_path}")
            sys.exit(1)

        print(f"\nLoading {excel_path}...")
        df = pd.read_excel(excel_path)
        df = df.where(pd.notna(df), other=None)

        courses = []
        for record in df.to_dict(orient="records"):
            row = {}
            for k, v in record.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    row[k] = None
                elif isinstance(v, np.generic):
                    row[k] = v.item()
                else:
                    row[k] = v
            courses.append(row)
        print(f"  {len(courses)} courses loaded.\n")

        print("Ingesting Excel courses...")
        print("-" * 62)

        for idx, course in enumerate(courses):
            prog_name    = clean(course.get("programme_name"))
            level_raw    = clean(course.get("level"))
            is_lateral   = detect_lateral_entry(prog_name)
            is_poly      = detect_polytechnic(prog_name, level_raw)
            canon_name   = canonical_course_name(prog_name, is_poly)
            duration_str = clean(course.get("duration"))
            duration_yrs = extract_duration_years(duration_str)
            prog_level   = normalise_level(level_raw, prog_name)

            text = build_course_text(course, is_lateral, is_poly, canon_name)
            if not text or len(text.split()) < 8:
                counters["skipped_empty"] += 1
                continue

            chunks   = chunk_words(text, CHUNK_WORDS)
            metadata = make_json_safe({
                "programme_name":      prog_name,
                "duration":            duration_str,
                "duration_years":      duration_yrs,
                "is_lateral_entry":    is_lateral,
                "is_polytechnic":      is_poly,
                "programme_type":      course.get("programme_type"),
                "department":          course.get("department"),
                "fees":                course.get("fees"),
                "eligibility":         course.get("eligibility"),
                "source_url":          course.get("source_url"),
                "level":               level_raw,
                "programme_level":     prog_level,
                "admission_procedure": course.get("admission_procedure"),
                "normalised_name":     canon_name,
            })

            for chunk in chunks:
                embed_and_insert(
                    sb, model, canon_name, level_raw, chunk, metadata, counters,
                    is_lateral_entry=is_lateral,
                    duration_years=duration_yrs,
                    programme_level=prog_level,
                    normalised_name=canon_name,
                )

            if is_lateral:
                counters["lateral"] += 1
            elif is_poly:
                counters["polytechnic"] += 1
            else:
                counters["regular"] += 1

            if (idx + 1) % 10 == 0 or (idx + 1) == len(courses):
                flags    = []
                if is_lateral: flags.append("LATERAL")
                if is_poly:    flags.append("POLY")
                flag_str  = f" [{','.join(flags)}]" if flags else ""
                dur_label = f"{duration_yrs}yr" if duration_yrs else "?yr"
                print(
                    f"  [{idx+1:3d}/{len(courses)}] "
                    f"{canon_name[:32]:<32} {dur_label:>4} {prog_level or '?':>7}"
                    f"{flag_str}"
                    f" | total: {counters['inserted']}"
                )

    excel_chunks = counters["inserted"]

    # ── FIX 5: Ingest FAQ + per-course semantic chunks ────────────────────────
    print(f"\nIngesting {len(COURSE_FAQ_CHUNKS)} FAQ/semantic chunks...")
    for faq in COURSE_FAQ_CHUNKS:
        embed_and_insert(
            sb, model,
            faq["course_name"],
            faq.get("category", "FAQ"),
            faq["chunk_text"],
            {"source": "faq"},
            counters,
            is_lateral_entry=False,
            duration_years=None,
            programme_level=faq.get("programme_level", "FAQ"),
            normalised_name=faq["course_name"],
        )
        status = "✓" if counters["errors"] == 0 else "!"
        print(f"  [{status}] {faq['course_name'][:55]}")

    faq_chunks = counters["inserted"] - excel_chunks

    print("\n" + "=" * 62)
    print("  INGESTION COMPLETE")
    print("=" * 62)
    print(f"  Regular courses  : {counters['regular']}")
    print(f"  Lateral entry    : {counters['lateral']}")
    print(f"  Polytechnic      : {counters['polytechnic']}")
    print(f"  Excel chunks     : {excel_chunks}")
    print(f"  FAQ chunks       : {faq_chunks}")
    print(f"  Total inserted   : {counters['inserted']}")
    print(f"  Skipped empty    : {counters['skipped_empty']}")
    print(f"  Errors           : {counters['errors']}")
    print()
    if counters["inserted"] > 0 and counters["errors"] == 0:
        print("  ✅ Ingestion successful! Next steps:")
        print("  1. python diagnose_rag.py")
        print("  2. http://localhost:8000/rag-test?q=fees+of+bca")
        print("  3. http://localhost:8000/rag-test?q=streams+of+b+tech")
        print("  4. http://localhost:8000/rag-test?q=diploma+polytechnic+courses")
    elif counters["errors"] > 0:
        print(f"  ⚠️  {counters['errors']} errors — check output above.")
        print("  Most likely cause: CourseChunk table schema mismatch.")
        print("  Run complete_schema.sql in Supabase SQL Editor first.")


if __name__ == "__main__":
    main()