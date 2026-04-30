"""
ingest_courses.py — Fixed ingestion for Invertis University
============================================================
FIXES vs previous version
──────────────────────────
FIX 1 — short_name() no longer splits on "." so "B.Sc.", "Ph.D.",
         "M.Tech." are preserved instead of being truncated to "B", "Ph", "M".

FIX 2 — parse_eligibility() now correctly separates fees from eligibility.
         Root cause: scraped data puts fees inside eligibility column.
         Previous version didn't strip fee amounts fully.

FIX 3 — build_course_text() now explicitly labels each section so the LLM
         can clearly distinguish "Fee structure" from "Eligibility criteria".

FIX 4 — --clear flag deletes old corrupted chunks before re-ingesting.
         Always run with --clear when re-ingesting after a data fix.

FIX 5 — programme_name is used as the primary searchable text block
         (it contains the real course description in your Excel).
         Previous version was ignoring it as a description field.

Run (ALWAYS use --clear when re-running):
  python ingest_courses.py --clear
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
CHUNK_WORDS     = 200   # smaller chunks = more precise retrieval


# ── University FAQ — covers hostel, campus, general info ─────────────────────
UNIVERSITY_FAQ_CHUNKS = [
    {
        "course_name": "Invertis University General Information",
        "category":    "FAQ",
        "chunk_text": (
            "Invertis University is located in Bareilly, Uttar Pradesh, India. "
            "The university offers undergraduate, postgraduate, diploma, and PhD programmes "
            "in Engineering, Management, Law, Agriculture, Pharmacy, Science, and Commerce. "
            "Invertis University is approved by UGC, AICTE, and BCI. "
            "The university conducts its own entrance test called IUCET "
            "(Invertis University Common Entrance Test). "
            "Admissions are also accepted based on JEE, JEECUP, CAT, MAT, CMAT, "
            "and merit of qualifying exam. "
            "Academic session 2025-26 admissions are open."
        ),
    },
    {
        "course_name": "Invertis University Hostel and Campus Facilities",
        "category":    "FAQ",
        "chunk_text": (
            "Invertis University has separate hostel facilities for boys and girls on campus. "
            "The hostels are equipped with furnished rooms, Wi-Fi internet, mess and canteen, "
            "laundry facility, and 24-hour security. "
            "The campus has a well-equipped library with thousands of books and digital resources. "
            "Sports facilities include cricket ground, basketball court, volleyball, and indoor games. "
            "The campus has modern laboratories, computer labs, seminar halls, and auditoriums. "
            "Medical facility and health centre are available on campus. "
            "Transport facility is available for day scholars from Bareilly city. "
            "Yes, hostel accommodation is available for all students at Invertis University."
        ),
    },
    {
        "course_name": "Invertis University Admission Process",
        "category":    "FAQ",
        "chunk_text": (
            "Admission process at Invertis University: "
            "Step 1 — Fill the online application form on the Invertis University website. "
            "Step 2 — Appear for IUCET (Invertis University Common Entrance Test) "
            "or submit valid scores of JEE, JEECUP, CAT, MAT. "
            "Step 3 — Attend counselling and document verification. "
            "Step 4 — Pay the admission fee and confirm your seat. "
            "Documents required: 10th marksheet, 12th marksheet, transfer certificate, "
            "migration certificate, passport photos, Aadhaar card, "
            "and category certificate if applicable. "
            "Admissions open for 2025-26 academic session."
        ),
    },
    {
        "course_name": "Invertis University Scholarships and Fee Payment",
        "category":    "FAQ",
        "chunk_text": (
            "Invertis University offers merit-based scholarships for outstanding students. "
            "Scholarships are available for students scoring above 75 percent in qualifying exam. "
            "UP government scholarships pre-matric and post-matric are applicable for eligible students. "
            "Fee can be paid semester-wise or annually. "
            "Fee payment modes: online transfer, demand draft, or cash at the accounts office. "
            "General fee for undergraduate engineering programmes is approximately Rs 35,000 per year. "
            "MBA and management programmes have different fee structures. "
            "Contact the scholarship cell for latest details and fee waivers."
        ),
    },
    {
        "course_name": "Invertis University Placements and Career",
        "category":    "FAQ",
        "chunk_text": (
            "Invertis University has an active Training and Placement Cell. "
            "Top recruiters visit campus from IT, banking, manufacturing, and management sectors. "
            "The placement cell organises campus drives, mock interviews, resume workshops, "
            "and aptitude training. "
            "Students from Engineering, MBA, and BCA have strong placement records. "
            "Internship opportunities are arranged through industry tie-ups. "
            "The university has MoUs with several companies for student training."
        ),
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def short_name(prog_name: str) -> str:
    """
    FIX 1: Extract a clean course name WITHOUT splitting on "."
    so "B.Sc.", "Ph.D.", "M.Tech." are preserved fully.

    Strategy: take text up to the first occurrence of " deals with",
    " is a", " is the", " provides", " focuses" — these signal
    where the description starts in your programme_name column.
    """
    if not prog_name:
        return "Unknown Course"

    # Patterns that signal start of description text
    desc_starters = [
        " deals with", " is a ", " is the ", " provides ", " focuses ",
        " involves ", " offers ", " covers ", " aims ", " prepares ",
        " equips ", " trains ", " includes ",
    ]
    name = prog_name
    for starter in desc_starters:
        idx = prog_name.lower().find(starter)
        if idx > 5:   # must have at least 5 chars before it
            name = prog_name[:idx].strip()
            break
    else:
        # Fallback: take first 80 chars but don't split on "."
        name = prog_name[:80].strip()

    return name[:100]


def is_fee_like(text: str) -> bool:
    """Returns True if a pipe-separated chunk looks like a fee amount."""
    fee_indicators = [
        r"\d+,\d{3}",    # 35,000 or 80,000
        r"\d+styr",      # 1StYr
        r"\d+ndyr",      # 2NdYr
        r"\d+rdyr",      # 3RdYr
        r"\d+thyr",      # 4ThYr
        r"per year",
        r"per annum",
    ]
    lower = text.lower()
    return any(re.search(pat, lower) for pat in fee_indicators)


def parse_fees(raw: str) -> str:
    """
    Converts '35,000 1StYr. | 35,000 2NdYr. | 35,000 3RdYr.'
    into clearly labeled lines.
    """
    if not raw:
        return ""

    if "|" not in raw:
        # Single fee value
        amount = re.sub(r"\d+styr\.?|\d+ndyr\.?|\d+rdyr\.?|\d+thyr\.?",
                        "", raw, flags=re.IGNORECASE).strip(" .")
        return f"Fee structure: Rs {amount} per year" if amount else ""

    parts = [p.strip() for p in raw.split("|") if p.strip()]
    year_labels = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]
    lines = []

    for i, part in enumerate(parts):
        # Remove year suffixes
        amount = re.sub(
            r"\d+styr\.?|\d+ndyr\.?|\d+rdyr\.?|\d+thyr\.?",
            "", part, flags=re.IGNORECASE
        ).strip(" .")
        label = year_labels[i] if i < len(year_labels) else f"Year {i+1}"
        if amount:
            lines.append(f"  {label}: Rs {amount}")

    return ("Fee structure (per year):\n" + "\n".join(lines)) if lines else ""


def parse_eligibility(raw: str) -> str:
    """
    FIX 2: Splits pipe-separated eligibility and aggressively filters
    out fee amounts that the scraper mixed into the eligibility column.
    """
    if not raw:
        return ""

    if "|" not in raw:
        # Single value — check if it's a fee
        return "" if is_fee_like(raw) else f"Eligibility: {raw}"

    parts = [p.strip() for p in raw.split("|") if p.strip()]
    eligibility_parts = [p for p in parts if not is_fee_like(p)]

    if not eligibility_parts:
        return ""

    lines = [f"  - {p}" for p in eligibility_parts]
    return "Eligibility criteria:\n" + "\n".join(lines)


def build_course_text(course: dict) -> str:
    """
    FIX 3: Builds a rich, clearly labeled text block.
    programme_name in your Excel contains the real description text.
    """
    parts: list[str] = []

    prog_name = clean(course.get("programme_name"))
    name      = short_name(prog_name) if prog_name else ""

    # Course identity — name + full description from programme_name
    if name:
        parts.append(f"Course: {name}")
    if prog_name and len(prog_name) > len(name) + 10:
        # The part after the short name is the description
        desc_start = len(name)
        description = prog_name[desc_start:].strip().lstrip(".").strip()
        if len(description) > 20:
            parts.append(f"About this course: {description}")

    # Classification
    dept  = clean(course.get("department"))
    level = clean(course.get("level"))
    ptype = clean(course.get("programme_type"))
    if dept:
        parts.append(f"Department: {dept}")
    if level:
        parts.append(f"Level: {level}")
    if ptype:
        parts.append(f"Programme type: {ptype}")

    # Duration
    duration = clean(course.get("duration"))
    if duration:
        parts.append(f"Duration: {duration}")

    # Fees
    fees_raw = clean(course.get("fees"))
    if fees_raw:
        fees_text = parse_fees(fees_raw)
        if fees_text:
            parts.append(fees_text)

    # Eligibility (FIX 2: filtered)
    elig_raw = clean(course.get("eligibility"))
    if elig_raw:
        elig_text = parse_eligibility(elig_raw)
        if elig_text:
            parts.append(elig_text)

    # Admission procedure
    proc = clean(course.get("admission_procedure"))
    if proc:
        if "|" in proc:
            proc_parts = [p.strip() for p in proc.split("|") if p.strip()]
            proc_lines = [f"  - {p}" for p in proc_parts]
            parts.append("Admission procedure:\n" + "\n".join(proc_lines))
        else:
            parts.append(f"Admission procedure: {proc}")

    return "\n\n".join(p for p in parts if p.strip())


def chunk_words(text: str, size: int = CHUNK_WORDS) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i: i + size]).strip()
        for i in range(0, len(words), size)
        if " ".join(words[i: i + size]).strip()
    ]


def embed_and_insert(sb, model, course_name: str, category: str,
                     chunk_text: str, metadata: dict, counters: dict) -> None:
    raw_emb   = model.encode(chunk_text)
    embedding = [float(x) for x in raw_emb]

    if any(math.isnan(x) or math.isinf(x) for x in embedding):
        counters["bad_embed"] += 1
        return

    payload = make_json_safe({
        "course_name": course_name,
        "category":    category or None,
        "chunk_text":  chunk_text,
        "embedding":   embedding,
        "metadata":    metadata,
    })

    try:
        sb.table("CourseChunk").insert(payload).execute()
        counters["inserted"] += 1
    except Exception as e:
        print(f"    [ERROR] Insert: {e}")
        counters["errors"] += 1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true",
                        help="Delete ALL old chunks before ingesting (recommended)")
    args = parser.parse_args()

    print("=" * 62)
    print("  Invertis University — Course Ingestion (Fixed)")
    print(f"  Model: {EMBEDDING_MODEL} | Chunk: {CHUNK_WORDS} words")
    print("=" * 62)

    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
    if not url or not key:
        print("[ERROR] Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env")
        sys.exit(1)

    excel_path = "data/final_invertis_courses.xlsx"
    if not os.path.exists(excel_path):
        print(f"[ERROR] Not found: {excel_path}")
        sys.exit(1)

    import httpx
    from supabase import create_client
    sb = create_client(url, key)
    sb.postgrest.session = httpx.Client(http2=False, timeout=30)
    print("Supabase connected.")

    # ── Clear old corrupted data ──────────────────────────────────────────────
    if args.clear:
        print("\nClearing ALL existing chunks (fixing corrupted data)...")
        try:
            result = sb.table("CourseChunk").select("id", count="exact").execute()
            old_count = result.count or 0
            sb.table("CourseChunk").delete().neq(
                "id", "00000000-0000-0000-0000-000000000000"
            ).execute()
            print(f"  Deleted {old_count} old chunks.")
        except Exception as e:
            print(f"  [WARN] Clear error: {e}")
    else:
        print("\n[TIP] Run with --clear to delete old corrupted chunks first.")

    # Load model
    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("  Ready.")

    # Load Excel
    print(f"Loading {excel_path}...")
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

    counters = {"inserted": 0, "bad_embed": 0, "errors": 0, "skipped_empty": 0}

    # ── Ingest course chunks ──────────────────────────────────────────────────
    print("Ingesting courses...")
    print("-" * 40)

    for idx, course in enumerate(courses):
        prog_name = clean(course.get("programme_name"))
        name      = short_name(prog_name) if prog_name else "Unknown"
        level     = clean(course.get("level"))

        text = build_course_text(course)
        if not text or len(text.split()) < 8:
            counters["skipped_empty"] += 1
            continue

        chunks   = chunk_words(text, CHUNK_WORDS)
        metadata = make_json_safe({
            "programme_name":      prog_name,
            "duration":            course.get("duration"),
            "programme_type":      course.get("programme_type"),
            "department":          course.get("department"),
            "fees":                course.get("fees"),
            "eligibility":         course.get("eligibility"),
            "source_url":          course.get("source_url"),
            "level":               level,
            "admission_procedure": course.get("admission_procedure"),
        })

        for chunk in chunks:
            embed_and_insert(sb, model, name, level, chunk, metadata, counters)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(courses):
            print(f"  [{idx+1:3d}/{len(courses)}] {name[:40]:<40} | total: {counters['inserted']}")

    course_chunks = counters["inserted"]

    # ── Ingest FAQ chunks ─────────────────────────────────────────────────────
    print("\nIngesting FAQ chunks...")
    for faq in UNIVERSITY_FAQ_CHUNKS:
        embed_and_insert(
            sb, model,
            faq["course_name"], faq["category"],
            faq["chunk_text"], {"source": "faq"},
            counters,
        )
        print(f"  ✓ {faq['course_name'][:55]}")

    faq_chunks = counters["inserted"] - course_chunks

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  DONE")
    print("=" * 62)
    print(f"  Course chunks : {course_chunks}")
    print(f"  FAQ chunks    : {faq_chunks}")
    print(f"  Total         : {counters['inserted']}")
    print(f"  Skipped empty : {counters['skipped_empty']}")
    print(f"  Errors        : {counters['errors']}")
    print()
    if counters["inserted"] > 0:
        print("  Next: python validate_rag.py")


if __name__ == "__main__":
    main()