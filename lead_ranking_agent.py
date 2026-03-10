"""
lead_listener.py — F2Fintech Lead Ranking Agent
================================================
Run this file on any machine (your colleague's laptop or a server).

It does TWO things:
  1. Listens for new leads inserted into Supabase via Realtime pub/sub.
  2. On every INSERT, immediately fetches scoring rules from Supabase,
     ranks the lead, and writes the result back to the Lead table
     (which the CRM frontend picks up instantly via its own Realtime sub).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SETUP (one-time, on your laptop)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Install dependencies:
       pip install realtime supabase httpx

  2. (Optional) To use a .env file instead of hard-coded keys, also run:
       pip install python-dotenv
     Then create a .env file next to this script with:
       SUPABASE_URL=...
       SUPABASE_SERVICE_KEY=...
       CRM_API_BASE=...

  3. Run the agent:
       python lead_listener.py

  4. Leave the terminal open. Every time a visitor submits the enquiry form
     on the website, you will see a log line here and the CRM will update
     within seconds.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WHICH KEY TO USE?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • The frontend (Next.js CRM) uses the ANON key  — safe for browsers.
  • This Python agent uses the SERVICE ROLE key   — never expose in browser.
    The service role key lets us:
      - Read the `scoring_rules` table
      - Write back aiScore / priority / nextBestAction to the Lead table
      - Bypass Row Level Security (RLS) so we don't need extra policies

  Get your SERVICE ROLE key from:
    Supabase Dashboard → Project Settings → API → service_role (secret)
"""

import asyncio
import os
from datetime import datetime, timezone

import httpx
from realtime import AsyncRealtimeClient
from supabase import create_client, Client

# ── Configuration ──────────────────────────────────────────────────────────────
# Either set the values directly here, OR use a .env file (see SETUP above).

# ✅ Supabase project URL — shared project for the whole team
SUPABASE_URL= ""

# ✅ Service role key — gives this agent write access + bypasses RLS.
#    Ask Faraz / check Supabase Dashboard → Project Settings → API
SUPABASE_SERVICE_KEY = ""

# Anon key — used ONLY for the Realtime WebSocket connection
#    (Realtime requires the anon key; service key doesn't work for it)
SUPABASE_ANON_KEY = ""

# CRM NestJS API — use the deployed URL in production,
#    or http://localhost:11000/api/v1 if Faraz's backend is running locally
CRM_API_BASE = ""
# CRM_API_BASE = "http://localhost:11000/api/v1"  # ← uncomment for local testing

POLL_INTERVAL_SECONDS = 5  # how often to check for new leads

# Allow overrides from environment variables / .env file
# make sure your .env contains SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_ANON_KEY
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — using hard-coded values above

SUPABASE_URL         = os.getenv("SUPABASE_URL",         SUPABASE_URL)
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY)
SUPABASE_ANON_KEY    = os.getenv("SUPABASE_ANON_KEY",    SUPABASE_ANON_KEY)
CRM_API_BASE         = os.getenv("CRM_API_BASE",         CRM_API_BASE)
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", POLL_INTERVAL_SECONDS))

# Validate required environment variables
if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL is required. Set it in .env file or environment variables.")
if not SUPABASE_SERVICE_KEY:
    raise ValueError("SUPABASE_SERVICE_KEY is required. Set it in .env file or environment variables.")
if not SUPABASE_ANON_KEY:
    raise ValueError("SUPABASE_ANON_KEY is required. Set it in .env file or environment variables.")
if not CRM_API_BASE:
    raise ValueError("CRM_API_BASE is required. Set it in .env file or environment variables.")

# ── Supabase REST client (for reads/writes, uses service key) ──────────────────
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ══════════════════════════════════════════════════════════════════════════════
#  RANKING HELPERS  (ported directly from your colleague's notebook)
# ══════════════════════════════════════════════════════════════════════════════

def get_priority(score: int) -> str:
    """Map numeric score → High / Medium / Low priority."""
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"


def get_next_action(priority: str) -> str:
    """Map priority string → recommended next action."""
    if priority == "High":
        return "Call & Close"
    elif priority == "Medium":
        return "Follow-up Call"
    else:
        return "Email Nurturing"


def predict_ltv(score: int) -> float:
    """
    Estimated lifetime value in INR.
    Simple linear model — replace with your ML formula if you have one.
    """
    base = 50_000.0
    return round(base * (score / 100) * 1.5, 2)


def rank_lead_with_rules(lead: dict) -> dict:
    """
    Fetches active rows from the `ScoringRule` Supabase table and applies
    them to the given lead dict.  Returns a dict with computed fields.

    ScoringRule table columns (as shown in Supabase):
        ruleType   TEXT  — 'source' | 'campaign' | 'tag'
        ruleKey    TEXT  — lowercase value to match (e.g. 'google ads')
        baseScore  INT4
        weight     NUMERIC
        active     BOOLEAN  (default: true)
        context    JSONB
        createdAt  TIMESTAMP
        updatedAt  TIMESTAMP

    Falls back to a sensible baseline if the table is empty or unreachable.
    """
    total_score = 0

    try:
        rules = (
            supabase.table("ScoringRule")
            .select("*")
            .eq("active", True)
            .execute()
            .data
        )
    except Exception as e:
        print(f"    ⚠ Could not fetch ScoringRule table: {e}")
        rules = []

    for rule in rules:
        
        rule_type = (rule.get("ruleType") or "").lower()
        key       = (rule.get("ruleKey") or "").lower()
        base      = rule.get("baseScore", 0)
        weight    = rule.get("weight", 1)

        if rule_type == "source":
            if (lead.get("source") or "").lower() == key:
                total_score += base * weight

        elif rule_type == "campaign":
            if (lead.get("campaign") or "").lower() == key:
                total_score += base * weight

        elif rule_type == "tag":
            tags = [t.lower() for t in (lead.get("tags") or [])]
            if key in tags:
                total_score += base * weight
    # If no rules matched at all, fall back to a simple field-based baseline
    if total_score == 0:
        total_score = 30  # baseline
        if (lead.get("source") or "").lower() in ("website", "google ads"):
            total_score += 20
        if lead.get("email"):
            total_score += 10
        if lead.get("phone"):
            total_score += 10

    final_score = int(max(0, min(total_score, 100)))
    priority    = get_priority(final_score)
    next_action = get_next_action(priority)
    ltv         = predict_ltv(final_score)

    return {
        "aiScore":        final_score,
        "score":          final_score,
        "priority":       priority,
        "nextBestAction": next_action,
        "predictedLTV":   ltv,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESS A SINGLE LEAD  (async — safe to await in the polling loop)
# ══════════════════════════════════════════════════════════════════════════════

async def process_lead(lead: dict):
    lead_id = lead.get("id")
    name    = lead.get("name") or "Unknown"
    course  = lead.get("course") or lead.get("company") or "—"

    print(f"\n[+] New lead: {name} | Course: {course} | ID: {lead_id}")
    print("LEAD:", lead)

    result = rank_lead_with_rules(lead)
    print(
        f"    → Score: {result['aiScore']}"
        f" | Priority: {result['priority']}"
        f" | Action: {result['nextBestAction']}"
        f" | LTV: ₹{result['predictedLTV']:,.0f}"
    )

    # Write back via NestJS PATCH endpoint
    try:
        async with httpx.AsyncClient(timeout=15) as http:
            response = await http.patch(
                f"{CRM_API_BASE}/update-lead/{lead_id}",
                json={
                    "aiScore":        result["aiScore"],
                    "score":          result["score"],
                    "priority":       result["priority"],
                    "nextBestAction": result["nextBestAction"],
                    "predictedLTV":   result["predictedLTV"],
                },
            )
            if response.status_code in (200, 201):
                print(f"    ✓ CRM updated successfully")
            else:
                print(f"    ✗ CRM PATCH failed: HTTP {response.status_code} — {response.text[:200]}")
    except Exception as exc:
        print(f"    ✗ Error contacting CRM API: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — POLLING LOOP
# ══════════════════════════════════════════════════════════════════════════════
async def main():
    print("=" * 60)
    print("  F2Fintech — Lead Ranking Agent (Polling Mode)")
    print(f"  Supabase : {SUPABASE_URL}")
    print(f"  CRM API  : {CRM_API_BASE}")
    print(f"  Interval : every {POLL_INTERVAL_SECONDS}s")

    try:
        rows = supabase.table("Lead").select("id").limit(1).execute()
        print(f"  → Supabase OK (rows visible: {len(rows.data or [])})")
    except Exception as e:
        print(f"  → Supabase FAILED: {e}")
        return

    print("=" * 60)
    print("[*] Polling for unranked leads... (Ctrl+C to stop)\n")

    last_checked = datetime.now(timezone.utc).isoformat()

    while True:
        try:
            now = datetime.now(timezone.utc).isoformat()

            rows = (
                supabase.table("Lead")
                .select("*")
                .gt("createdAt", last_checked)  # only leads newer than last poll
                .eq("aiScore", 0)               # not yet ranked
                .execute()
                .data
            ) or []

            last_checked = now

            if rows:
                print(f"[~] Found {len(rows)} new unranked lead(s)")
                for lead in rows:
                    await process_lead(lead)   # ✅ async, no sync/await mismatch

        except Exception as e:
            
            import traceback
            traceback.print_exc()

        await asyncio.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] Shutting down.")
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
