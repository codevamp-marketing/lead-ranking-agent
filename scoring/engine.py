"""
scoring/engine.py — Lead Scoring Engine
========================================

SCORING PHILOSOPHY
──────────────────
A lead score is built from two independent layers:

  Layer 1 — Rule-based table scores (admin-configurable via ScoringRule table)
             Source, Campaign, and Tag rules fetched from DB.
             Admin can tune these in the CRM portal without touching code.

  Layer 2 — Signal-based heuristic scores (hardcoded domain knowledge)
             These capture intent/quality signals from the lead's profile
             that no admin-configurable rule can easily express:
               • Course demand tier (MBA/PGDM rank higher than generic)
               • Specialization completeness (filled = more serious)
               • Contact info completeness (phone AND email = serious)
               • Attribution data (UTM keyword / gclid presence = paid intent)
               • Source quality multiplier (Google Ads > Referral > Manual etc.)

Both layers are additive.  Final score is clamped to [0, 100].

SCORING TABLE REFERENCE (Layer 2 defaults — matches your LeadSource enum)
──────────────────────────────────────────────────────────────────────────
  Source            Base
  ──────────────    ─────
  Google_Ads         25    ← High commercial intent (paid click)
  Facebook_Ads       18    ← Broad awareness — lower intent
  Instagram          15
  LinkedIn           20    ← Professional, B2B adjacent
  Website            20    ← Organic search = high intent
  Referral           22    ← Trust signal — closes faster
  Manual             10    ← Entered by counsellor — unknown quality
  (missing)           5    ← No source = poor data hygiene

  Course demand tier
  ──────────────────
  Tier-1 (MBA, PGDM, MCA, M.Tech)  → +20
  Tier-2 (BBA, BCA, B.Tech, etc.)  → +12
  Other / blank                     →  +5

  Contact completeness
  ────────────────────
  phone AND email     → +15
  phone only          → +10
  email only          →  +5
  neither             →   0

  Specialization filled  → +8
  Attribution UTM/gclid  → +5   (paid-ad confirmation)
"""

from __future__ import annotations

# ── Course demand tiers ───────────────────────────────────────────────────────
_TIER1_COURSES = {"mba", "pgdm", "mca", "m.tech", "mtech", "msc", "m.sc", "pgpm"}
_TIER2_COURSES = {"bba", "bca", "b.tech", "btech", "bsc", "b.sc", "bcom", "b.com", "ba"}

# ── Source quality scores (matches LeadSource enum exactly) ──────────────────
_SOURCE_SCORE: dict[str, int] = {
    "google_ads":    25,
    "facebook_ads":  18,
    "instagram":     15,
    "linkedin":      20,
    "website":       20,
    "referral":      22,
    "manual":        10,
}


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 1: Rule-based scoring (ScoringRule table)
# ══════════════════════════════════════════════════════════════════════════════

def _normalize(val: str) -> str:
    """
    Canonical form for matching: lowercase, underscores → spaces, strip.
    Applied to BOTH the ruleKey and the lead field value so matching is
    consistent regardless of how each side was entered.

    Examples:
      "Google_Ads"   → "google ads"
      "google ads"   → "google ads"   ← your ScoringRule ruleKey format
      "Google Ads"   → "google ads"
      "FACEBOOK_ADS" → "facebook ads"
    """
    return val.lower().replace("_", " ").strip()


def _apply_db_rules(lead: dict, rules: list[dict]) -> int:
    """
    Iterates active ScoringRule rows and returns total matched score.

    Your ScoringRule table columns (camelCase — Supabase REST returns as-is):
        ruleType  TEXT  — "source" | "campaign" | "tag"
        ruleKey   TEXT  — e.g. "google ads", "facebook", "python bootcamp"
        baseScore INT   — raw point value
        weight    NUM   — multiplier (e.g. 1.4)
        active    BOOL

    Matching uses _normalize() on BOTH sides — handles Google_Ads vs
    "google ads" vs "Google Ads" transparently.
    """
    total = 0
    for rule in rules:
        rule_type = (rule.get("ruleType") or "").lower().strip()
        key       = _normalize(rule.get("ruleKey") or "")
        base      = int(rule.get("baseScore") or 0)
        weight    = float(rule.get("weight") or 1.0)

        if not key or base == 0:
            continue   # skip malformed rules silently

        if rule_type == "source":
            # Lead source comes from LeadSource enum: "Google_Ads", "Facebook_Ads" etc.
            source = _normalize(lead.get("source") or "")
            if key in source or source in key:   # bidirectional — handles partial keys
                total += int(base * weight)

        elif rule_type == "campaign":
            campaign = _normalize(lead.get("campaign") or "")
            if key in campaign:
                total += int(base * weight)

        elif rule_type == "tag":
            tags = [_normalize(t) for t in (lead.get("tags") or []) if t]
            if any(key in t or t in key for t in tags):
                total += int(base * weight)

    return total


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 2: Signal-based heuristic scoring
# ══════════════════════════════════════════════════════════════════════════════

def _source_score(lead: dict) -> int:
    # Normalize: "Google_Ads" → "google_ads", "google ads" → "google_ads"
    raw = _normalize(lead.get("source") or "").replace(" ", "_")
    return _SOURCE_SCORE.get(raw, 5)


def _course_score(lead: dict) -> int:
    course = (lead.get("course") or "").lower().strip()
    if not course:
        return 5
    if any(t in course for t in _TIER1_COURSES):
        return 20
    if any(t in course for t in _TIER2_COURSES):
        return 12
    return 8


def _contact_score(lead: dict) -> int:
    has_phone = bool((lead.get("phone") or "").strip())
    has_email = bool((lead.get("email") or "").strip())
    if has_phone and has_email:
        return 15
    if has_phone:
        return 10
    if has_email:
        return 5
    return 0


def _specialization_score(lead: dict) -> int:
    return 8 if (lead.get("specialization") or "").strip() else 0


def _attribution_score(lead: dict) -> int:
    """
    Attribution JSON may contain: gclid, fbclid, utm_source, utm_campaign,
    utm_medium, landing_page, keyword.
    Presence of gclid/fbclid confirms a paid-ad click — high intent signal.
    """
    attr = lead.get("attribution") or {}
    if isinstance(attr, str):
        try:
            import json
            attr = json.loads(attr)
        except Exception:
            attr = {}
    score = 0
    if attr.get("gclid") or attr.get("fbclid"):
        score += 5   # Confirmed paid ad click
    if attr.get("utm_campaign"):
        score += 2   # Campaign-tagged traffic
    return score


def _signal_score(lead: dict) -> int:
    return (
        _source_score(lead)
        + _course_score(lead)
        + _contact_score(lead)
        + _specialization_score(lead)
        + _attribution_score(lead)
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def score_lead(lead: dict, rules: list[dict]) -> int:
    """
    Combines Layer 1 (DB rules) + Layer 2 (signal heuristics).
    If DB rules produce nothing (empty table or no matches), Layer 2 alone
    drives the score — agent never returns a misleading 0.
    Result is clamped to [0, 100].
    """
    rule_score   = _apply_db_rules(lead, rules)
    signal_score = _signal_score(lead)

    # If admin has configured rules AND they matched, blend both layers.
    # If no rules matched, rely entirely on signal scoring.
    if rule_score > 0:
        # Blend: 60% rule-based, 40% signal — rules are intentional admin config
        combined = int(rule_score * 0.6 + signal_score * 0.4)
    else:
        combined = signal_score

    return max(0, min(combined, 100))


def classify_lead(score: int) -> str:
    """
    Hot  ≥ 60  — High priority, immediate call
    Warm ≥ 35  — Moderate priority, follow-up call
    Cold < 35  — Low priority, nurture sequence
    Thresholds validated against LeadType enum: Hot | Warm | Cold
    """
    if score >= 60:
        return "Hot"
    elif score >= 35:
        return "Warm"
    return "Cold"


def next_best_action(lead_type: str, lead: dict) -> str:
    """
    Returns the recommended next action string for the counsellor.
    Uses lead_type as primary driver, but enriches with lead context.
    """
    has_phone = bool((lead.get("phone") or "").strip())

    if lead_type == "Hot":
        return "Call & Close — Call within 1 hour" if has_phone else "Email Proposal Immediately"
    elif lead_type == "Warm":
        return "Follow-up Call — Schedule within 24 hours" if has_phone else "Send Course Brochure via Email"
    else:
        # Cold — enroll in drip nurture sequence
        return "Email Nurturing — Enroll in 7-day drip sequence"


def predict_ltv(score: int, lead: dict) -> float:
    """
    Estimated lifetime value (fee revenue) in INR.

    Formula:
      base_fee × course_multiplier × conversion_probability

    base_fee            = ₹2,00,000 (conservative average program fee)
    course_multiplier   = 1.5 for Tier-1, 1.0 for Tier-2, 0.7 for others
    conversion_prob     = score / 100
    """
    BASE_FEE = 200_000.0

    course = (lead.get("course") or "").lower()
    if any(t in course for t in _TIER1_COURSES):
        multiplier = 1.5
    elif any(t in course for t in _TIER2_COURSES):
        multiplier = 1.0
    else:
        multiplier = 0.7

    ltv = BASE_FEE * multiplier * (score / 100)
    return round(ltv, 2)