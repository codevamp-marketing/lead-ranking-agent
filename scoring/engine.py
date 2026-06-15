"""
scoring/engine.py — Lead Scoring Engine
========================================

SCORING PHILOSOPHY
──────────────────
A lead score is built from two independent layers:

  Layer 1 — Rule-based table scores (admin-configurable via ScoringRule table)
             Source, Campaign, and Tag rules fetched from DB.

  Layer 2 — Signal-based heuristic scores (hardcoded domain knowledge)
             Course demand tier, contact completeness, attribution signals.

Both layers are additive. Final score is clamped to [0, 100].

FIXES (v2.1)
────────────
  • _apply_db_rules: removed bidirectional "source in key" match.
    The old logic matched "google" against "google_ads" (false positive).
    Now only "key in source" is used (rule key must be a substring of
    the lead's normalised source value), which is directional and safe.
  • Blending ratio is now documented explicitly with rationale.
"""

from __future__ import annotations
import json

# ── Course demand tiers ───────────────────────────────────────────────────────
_TIER1_COURSES = {"mba", "pgdm", "mca", "m.tech", "mtech", "msc", "m.sc", "pgpm"}
_TIER2_COURSES = {"bba", "bca", "b.tech", "btech", "bsc", "b.sc", "bcom", "b.com", "ba"}

# ── Source quality scores (matches LeadSource enum exactly) ──────────────────
_SOURCE_SCORE: dict[str, int] = {
    "google_ads":   25,
    "facebook_ads": 18,
    "instagram":    15,
    "linkedin":     20,
    "website":      20,
    "referral":     22,
    "manual":       10,
}


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 1: Rule-based scoring (ScoringRule table)
# ══════════════════════════════════════════════════════════════════════════════

def _normalize(val: str) -> str:
    """
    Canonical form for matching: lowercase, underscores → spaces, strip.

    Examples:
      "Google_Ads"   → "google ads"
      "FACEBOOK_ADS" → "facebook ads"
      "google ads"   → "google ads"
    """
    return val.lower().replace("_", " ").strip()


def _apply_db_rules(lead: dict, rules: list[dict]) -> int:
    """
    Iterates active ScoringRule rows and returns total matched score.

    ScoringRule table columns:
        ruleType  TEXT  — "source" | "campaign" | "tag"
        ruleKey   TEXT  — e.g. "google ads", "facebook", "python bootcamp"
        baseScore INT   — raw point value
        weight    NUM   — multiplier (e.g. 1.4)
        active    BOOL

    FIX: matching is now unidirectional (key in field_value).
    The old bidirectional check (key in source OR source in key) caused
    false positives: a lead with source="google" matched rule key="google_ads".
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
            source = _normalize(lead.get("source") or "")
            # FIX: unidirectional — rule key must appear inside lead's source string
            if key in source:
                total += int(base * weight)

        elif rule_type == "campaign":
            campaign = _normalize(lead.get("campaign") or "")
            if key in campaign:
                total += int(base * weight)

        elif rule_type == "tag":
            tags = [_normalize(t) for t in (lead.get("tags") or []) if t]
            if any(key in t for t in tags):
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
    Attribution JSON may contain: gclid, fbclid, utm_source, utm_campaign.
    gclid/fbclid confirms a paid-ad click — high intent signal.
    """
    attr = lead.get("attribution") or {}
    if isinstance(attr, str):
        try:
            attr = json.loads(attr)
        except Exception:
            attr = {}
    score = 0
    if attr.get("gclid") or attr.get("fbclid"):
        score += 5
    if attr.get("utm_campaign"):
        score += 2
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

    Blending rationale:
      When admin rules are configured and matching, they represent
      intentional business logic and should dominate (60%).
      Signal heuristics provide a floor and refinement (40%).
      If no rules match, signal score alone drives the result.

    Result is clamped to [0, 100].
    """
    rule_score   = _apply_db_rules(lead, rules)
    signal_score = _signal_score(lead)

    if rule_score > 0:
        combined = int(rule_score * 0.6 + signal_score * 0.4)
    else:
        combined = signal_score

    return max(0, min(combined, 100))


def classify_lead(score: int) -> str:
    """
    Hot  ≥ 60  — High priority, immediate call
    Warm ≥ 35  — Moderate priority, follow-up call
    Cold < 35  — Low priority, nurture sequence
    """
    if score >= 60:
        return "Hot"
    elif score >= 35:
        return "Warm"
    return "Cold"


def next_best_action(lead_type: str, lead: dict) -> str:
    """Returns the recommended next action string for the counsellor."""
    has_phone = bool((lead.get("phone") or "").strip())

    if lead_type == "Hot":
        return "Call & Close — Call within 1 hour" if has_phone else "Email Proposal Immediately"
    elif lead_type == "Warm":
        return "Follow-up Call — Schedule within 24 hours" if has_phone else "Send Course Brochure via Email"
    else:
        return "Email Nurturing — Enroll in 7-day drip sequence"


def predict_ltv(score: int, lead: dict) -> float:
    """
    Estimated lifetime value (fee revenue) in INR.

    Formula:  base_fee × course_multiplier × conversion_probability
      base_fee           = ₹2,00,000 (conservative average program fee)
      course_multiplier  = 1.5 for Tier-1, 1.0 for Tier-2, 0.7 for others
      conversion_prob    = score / 100
    """
    BASE_FEE = 200_000.0

    course = (lead.get("course") or "").lower()
    if any(t in course for t in _TIER1_COURSES):
        multiplier = 1.5
    elif any(t in course for t in _TIER2_COURSES):
        multiplier = 1.0
    else:
        multiplier = 0.7

    return round(BASE_FEE * multiplier * (score / 100), 2)