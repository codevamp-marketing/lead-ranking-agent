"""
lead_ranking_agent.py — F2Fintech Lead Ranking Agent (LISTEN/NOTIFY edition)
=============================================================================
Upgraded from polling → instant Postgres LISTEN/NOTIFY.

How it works:
  1. Connects to Supabase Postgres directly via psycopg2.
  2. Runs LISTEN new_lead — a pg_notify fires on every Lead INSERT (triggered by SQL).
  3. Receives the full lead JSON instantly, scores it, and PATCHes back to the CRM.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SETUP (one-time)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Install dependencies:
       pip install -r requirements.txt

  2. Fill in .env (copy the DATABASE_URL from Supabase Dashboard →
     Settings → Database → Connection String → URI, Session mode, port 5432):
       DATABASE_URL=postgresql://postgres.[ref]:[password]@aws-...supabase.com:5432/postgres

  3. Run:
       python lead_ranking_agent.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SQL TRIGGER (run once in Supabase SQL Editor)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  See README.md or the implementation_plan.md for the SQL.
"""

import asyncio
import json
import os
import select

import httpx
import psycopg2
from supabase import create_client, Client

# ── Configuration ─────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — use environment variables directly

SUPABASE_URL         = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
CRM_API_BASE         = os.environ.get("CRM_API_BASE", "")
DATABASE_URL         = os.environ.get("DATABASE_URL", "")

# Validate required variables
_missing = [k for k, v in {
    "SUPABASE_URL":         SUPABASE_URL,
    "SUPABASE_SERVICE_KEY": SUPABASE_SERVICE_KEY,
    "CRM_API_BASE":         CRM_API_BASE,
    "DATABASE_URL":         DATABASE_URL,
}.items() if not v]

if _missing:
    raise ValueError(
        f"Missing required environment variables: {', '.join(_missing)}\n"
        "Set them in your .env file."
    )

# Supabase REST client — used for ScoringRule reads (uses service key → bypasses RLS)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ══════════════════════════════════════════════════════════════════════════════
#  RANKING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_priority(score: int) -> str:
    """Map numeric score → High / Medium / Low."""
    if score >= 70:   return "High"
    elif score >= 40: return "Medium"
    else:             return "Low"


def get_next_action(priority: str) -> str:
    return {
        "High":   "Call & Close",
        "Medium": "Follow-up Call",
        "Low":    "Email Nurturing",
    }[priority]


def predict_ltv(score: int) -> float:
    """Simple linear LTV estimate in INR. Replace with ML formula if available."""
    return round(50_000.0 * (score / 100) * 1.5, 2)


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
        print(f"    ⚠ ScoringRule fetch failed: {e}")
        rules = []

    for rule in rules:
        rule_type = (rule.get("ruleType") or "").lower()
        key       = (rule.get("ruleKey")  or "").lower()
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

    # Baseline fallback when no rules matched
    if total_score == 0:
        total_score = 30
        if (lead.get("source") or "").lower() in ("website", "google_ads", "google ads"):
            total_score += 20
        if lead.get("email"):
            total_score += 10
        if lead.get("phone"):
            total_score += 10

    final_score = int(max(0, min(total_score, 100)))
    priority    = get_priority(final_score)

    return {
        "aiScore":        final_score,
        "score":          final_score,
        "priority":       priority,
        "nextBestAction": get_next_action(priority),
        "predictedLTV":   predict_ltv(final_score),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESS A SINGLE LEAD
# ══════════════════════════════════════════════════════════════════════════════

async def process_lead(lead: dict):
    lead_id = lead.get("id")
    name    = lead.get("name") or "Unknown"
    course  = lead.get("course") or lead.get("company") or "—"

    print(f"\n[+] New lead: {name}  |  Course: {course}  |  ID: {lead_id}")

    result = rank_lead_with_rules(lead)
    print(
        f"    → Score: {result['aiScore']}"
        f" | Priority: {result['priority']}"
        f" | Action: {result['nextBestAction']}"
        f" | LTV: ₹{result['predictedLTV']:,.0f}"
    )

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
                print(f"    ✓ CRM updated — HTTP {response.status_code}")
            else:
                print(f"    ✗ CRM PATCH failed — HTTP {response.status_code}: {response.text[:300]}")
    except Exception as exc:
        print(f"    ✗ Error contacting CRM API: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
#  LISTEN / NOTIFY LOOP  (replaces the old polling loop)
# ══════════════════════════════════════════════════════════════════════════════

async def listen_loop():
    """
    Opens a persistent psycopg2 connection, issues LISTEN new_lead,
    and processes every notification that Postgres sends.

    AUTOCOMMIT must be set (isolation_level=0) — LISTEN/NOTIFY does not
    work inside a transaction block.
    """
    loop = asyncio.get_event_loop()

    print("[*] Connecting to Postgres...")
    conn = await loop.run_in_executor(
        None,
        lambda: psycopg2.connect(DATABASE_URL)
    )
    conn.set_isolation_level(0)  # AUTOCOMMIT — required for LISTEN
    cur = conn.cursor()
    cur.execute("LISTEN new_lead;")
    print("[✓] Listening on channel: new_lead")
    print("[*] Waiting for new leads... (Ctrl+C to stop)\n")

    while True:
        try:
            # Block up to 5s waiting for a notification (non-blocking feel in async)
            await loop.run_in_executor(
                None,
                lambda: select.select([conn], [], [], 5)
            )
            conn.poll()

            while conn.notifies:
                notify = conn.notifies.pop(0)
                try:
                    lead = json.loads(notify.payload)
                    await process_lead(lead)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Failed to parse lead JSON: {e}")
                except Exception as e:
                    import traceback
                    print(f"[ERROR] Exception while processing lead:")
                    traceback.print_exc()

        except psycopg2.OperationalError as e:
            # Connection dropped — attempt reconnect after 5s
            print(f"\n[!] Postgres connection lost: {e}")
            print("[*] Reconnecting in 5 seconds...")
            await asyncio.sleep(5)
            try:
                conn = await loop.run_in_executor(
                    None,
                    lambda: psycopg2.connect(DATABASE_URL)
                )
                conn.set_isolation_level(0)
                cur = conn.cursor()
                cur.execute("LISTEN new_lead;")
                print("[✓] Reconnected and listening on: new_lead\n")
            except Exception as reconnect_err:
                print(f"[FATAL] Reconnect failed: {reconnect_err}")
                raise


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("  F2Fintech — Lead Ranking Agent (LISTEN/NOTIFY Mode)")
    print(f"  CRM API  : {CRM_API_BASE}")
    print(f"  Supabase : {SUPABASE_URL}")
    print("=" * 60)

    # Quick sanity check — verify Supabase REST is reachable
    try:
        rows = supabase.table("Lead").select("id").limit(1).execute()
        print(f"[✓] Supabase REST OK (rows visible: {len(rows.data or [])})")
    except Exception as e:
        print(f"[!] Supabase REST check failed: {e}")

    await listen_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] Stopped by user.")
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
