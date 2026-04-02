"""
lead_ranking_agent.py  —  F2Fintech Lead Ranking Agent  (Production-Grade)
===========================================================================

Architecture overview
─────────────────────
  • Postgres LISTEN/NOTIFY  →  zero-latency lead arrival detection
  • asyncio.create_task()   →  each lead processed concurrently, listener never blocked
  • Rule-based scoring      →  reads ScoringRule table (source / campaign / tag)
  • Signal-based scoring    →  enriches score from lead profile fields
  • Tiered classification   →  Hot / Warm / Cold  +  nextBestAction
  • CRM PATCH w/ retry      →  exponential backoff + jitter, 3 attempts
  • Supabase fallback        →  if CRM exhausts retries, writes direct to DB
  • Dead-letter queue (DLQ) →  in-memory queue for leads that fail all retries
  • DLQ reprocessor         →  background task retries DLQ every 60s
  • Activity log            →  AI_Insight row for full audit trail
  • Notification            →  counsellor alert on ownerId
  • Structured JSON logging →  Datadog / CloudWatch ready

Concurrency model — WHY create_task matters
────────────────────────────────────────────
  The listener loop runs in a single thread. If process_lead() were awaited
  directly, Lead #2 would block behind Lead #1's entire pipeline (score +
  CRM PATCH + retries). With create_task(), the event loop interleaves all
  in-flight leads concurrently — the listener is free to drain notifies
  immediately, regardless of how many leads are mid-processing.

  Timeline (correct):
    t=0s  Lead #1 arrives → create_task(process_lead #1)  ← returns instantly
    t=0s  Lead #2 arrives → create_task(process_lead #2)  ← returns instantly
    t=0s  Listener back to select() — zero blocking
    t=1s  Lead #1 CRM fails → retry 1 (2s backoff) — event loop free
    t=1s  Lead #2 CRM succeeds → done
    t=3s  Lead #1 retry 2 …

Dead-letter queue
──────────────────
  If all 3 CRM retries + Supabase direct write fail, the scored payload is
  pushed to DLQ (in-memory list). A background task polls DLQ every 60s and
  retries. This survives transient infrastructure outages without losing data.

Run
────
  python -m agent.lead_ranking_agent
"""

import sys
import os

# Ensure repo root is on sys.path so package imports resolve correctly
# regardless of how the agent is invoked:
#   python -m agent.lead_ranking_agent   (local / Docker recommended)
#   python agent/lead_ranking_agent.py   (direct script)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import asyncio
import json
import random
import select
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import httpx
import psycopg2
from supabase import create_client, Client

from scoring.engine import score_lead, classify_lead, predict_ltv, next_best_action
from utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  DEAD-LETTER QUEUE
#  In-memory — survives transient failures, not agent restarts.
#  For persistence across restarts, swap deque for a Redis list or DB table.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DLQEntry:
    lead_id:   str
    payload:   dict          # scored fields ready to write
    attempts:  int = 0
    queued_at: float = field(default_factory=time.time)

_dlq: deque[DLQEntry] = deque(maxlen=1000)   # cap at 1000 entries


# ══════════════════════════════════════════════════════════════════════════════
#  SUPABASE CLIENT
#  StreamReset (error_code:1) is caused by HTTP/2 stream resets on some
#  Supabase regions. ClientOptions with explicit timeouts stabilises the
#  connection. If StreamReset persists, also run:
#    pip install httpx[http2] --upgrade
# ══════════════════════════════════════════════════════════════════════════════

try:
    from supabase.lib.client_options import ClientOptions
    supabase: Client = create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_SERVICE_KEY,
        options=ClientOptions(
            postgrest_client_timeout=10,
            storage_client_timeout=10,
        )
    )
except Exception:
    # Fallback for older supabase-py versions without ClientOptions timeout params
    supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDE-EFFECT WRITERS  (Activity log + Notification)
# ══════════════════════════════════════════════════════════════════════════════

def _log_ai_activity(lead_id: str, score: int, lead_type: str, action: str, ltv: float):
    """
    Inserts an Activity row of type AI_Insight so counsellors can see
    exactly what the agent decided and why.
    """
    description = (
        f"AI ranked this lead as '{lead_type}' (score {score}/100). "
        f"Predicted LTV: ₹{ltv:,.0f}. Recommended action: {action}."
    )
    try:
        supabase.table("Activity").insert({
            "leadId":      lead_id,
            "type":        "AI_Insight",
            "description": description,
            "metadata": {
                "aiScore":        score,
                "type":           lead_type,
                "nextBestAction": action,
                "predictedLTV":   ltv,
            },
        }).execute()
    except Exception as exc:
        logger.warning("activity_log_failed", lead_id=lead_id, error=str(exc))


def _send_notification(lead: dict, lead_type: str, action: str):
    """
    Creates an in-app Notification for the assigned counsellor.

    Schema uses `pickedBy` (String?) — the counsellor who claimed this lead.
    If not yet picked, no notification is sent — one will fire when a
    counsellor picks the lead via the CRM Kanban board.
    """
    # Your updated schema: ownerId removed, counsellor identified by pickedBy
    counsellor_id = lead.get("pickedBy") or lead.get("createdBy")
    if not counsellor_id:
        return   # No counsellor assigned yet — skip silently

    lead_id   = lead.get("id")
    lead_name = lead.get("name") or "A new lead"

    if lead_type == "Hot":
        title = f"🔥 Hot Lead: {lead_name} — Call Now"
    elif lead_type == "Warm":
        title = f"🌡 Warm Lead: {lead_name} — Follow Up"
    else:
        title = f"❄ Cold Lead: {lead_name} — Nurture"

    body = f"AI Score: {lead.get('aiScore', '—')} | Action: {action}"

    try:
        supabase.table("notifications").insert({
            "userId":  counsellor_id,
            "leadId":  lead_id,
            "title":   title,
            "body":    body,
            "read":    False,
        }).execute()
        logger.info("notification_sent", lead_id=lead_id, counsellor_id=counsellor_id)
    except Exception as exc:
        logger.warning("notification_failed", lead_id=lead_id, error=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
#  CRM PATCH  —  exponential backoff + jitter
# ══════════════════════════════════════════════════════════════════════════════

_MAX_ATTEMPTS   = 3
_BASE_DELAY_SEC = 2.0   # first retry after ~2s
_MAX_DELAY_SEC  = 16.0  # cap so we don't wait forever


async def _patch_crm(lead_id: str, payload: dict) -> bool:
    """
    Attempts CRM PATCH with exponential backoff + full jitter.

    Retry schedule (approximate):
      Attempt 1 → immediate
      Attempt 2 → wait  2–4s  (base × 2^0 + jitter)
      Attempt 3 → wait  4–8s  (base × 2^1 + jitter)

    Returns True only on HTTP 200.  Does NOT block the listener — this
    function is always called inside an asyncio.Task (create_task), so
    the await asyncio.sleep() yields control back to the event loop,
    letting other leads process concurrently during the wait.
    """
    url = f"{settings.CRM_API_BASE}/update-lead/{lead_id}"

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=settings.CRM_TIMEOUT) as http:
                resp = await http.patch(url, json=payload)
                if resp.status_code == 200:
                    if attempt > 1:
                        logger.info("crm_patch_ok_after_retry", lead_id=lead_id, attempt=attempt)
                    return True
                logger.warning(
                    "crm_patch_bad_status",
                    lead_id=lead_id, attempt=attempt,
                    status=resp.status_code, body=resp.text[:200],
                )

        except httpx.ConnectError:
            logger.warning("crm_unreachable", lead_id=lead_id, attempt=attempt, url=url)
        except httpx.TimeoutException:
            logger.warning("crm_timeout", lead_id=lead_id, attempt=attempt)
        except Exception as exc:
            logger.warning("crm_error", lead_id=lead_id, attempt=attempt, error=str(exc))

        if attempt < _MAX_ATTEMPTS:
            # Exponential backoff with full jitter:
            #   sleep = random(0, min(cap, base × 2^(attempt-1)))
            # Jitter prevents thundering herd if many leads fail simultaneously.
            ceiling = min(_MAX_DELAY_SEC, _BASE_DELAY_SEC * (2 ** (attempt - 1)))
            delay   = random.uniform(0, ceiling)
            logger.info("crm_retry_backoff", lead_id=lead_id, attempt=attempt, wait_sec=round(delay, 2))
            await asyncio.sleep(delay)   # ← yields to event loop; other tasks run here

    logger.error("crm_all_retries_exhausted", lead_id=lead_id)
    return False


async def _write_scored_payload(lead_id: str, payload: dict) -> bool:
    """
    Two-tier write:
      1. Try CRM REST API (with internal retries).
      2. If CRM fails, fall back to Supabase direct write.
      3. If both fail, push to DLQ for background reprocessing.

    Always returns True if data was written *somewhere* durable.
    """
    # Tier 1 — CRM API
    if await _patch_crm(lead_id, payload):
        return True

    # Tier 2 — Supabase direct (bypasses CRM business logic, but data is safe)
    logger.warning("crm_exhausted_trying_supabase_direct", lead_id=lead_id)
    try:
        supabase.table("Lead").update(payload).eq("id", lead_id).execute()
        logger.info("supabase_direct_write_ok", lead_id=lead_id)
        return True
    except Exception as exc:
        logger.error("supabase_direct_write_failed", lead_id=lead_id, error=str(exc))

    # Tier 3 — DLQ (background reprocessor will retry)
    _dlq.append(DLQEntry(lead_id=lead_id, payload=payload))
    logger.error("lead_pushed_to_dlq", lead_id=lead_id, dlq_size=len(_dlq))
    return False


async def _dlq_reprocessor():
    """
    Background task: wakes every 60s, drains DLQ entries one by one.
    Only retries via Supabase direct (CRM was already exhausted).
    Entries that succeed are removed; persistent failures stay in DLQ
    and are logged so ops can investigate.
    """
    logger.info("dlq_reprocessor_started")
    while True:
        await asyncio.sleep(60)
        if not _dlq:
            continue

        logger.info("dlq_reprocessor_tick", pending=len(_dlq))
        # Snapshot current length so new arrivals during processing aren't
        # touched this cycle.
        batch = [_dlq.popleft() for _ in range(len(_dlq))]

        for entry in batch:
            entry.attempts += 1
            try:
                supabase.table("Lead").update(entry.payload).eq("id", entry.lead_id).execute()
                logger.info("dlq_reprocess_ok", lead_id=entry.lead_id, attempts=entry.attempts)
            except Exception as exc:
                logger.error(
                    "dlq_reprocess_failed",
                    lead_id=entry.lead_id, attempts=entry.attempts, error=str(exc),
                )
                if entry.attempts < 10:          # give up after 10 total attempts (~10 min)
                    _dlq.append(entry)
                else:
                    logger.critical(
                        "dlq_entry_abandoned",
                        lead_id=entry.lead_id,
                        payload=entry.payload,
                        queued_at=entry.queued_at,
                    )


# ══════════════════════════════════════════════════════════════════════════════
#  CORE: PROCESS ONE LEAD
# ══════════════════════════════════════════════════════════════════════════════

async def process_lead(lead: dict):
    lead_id = lead.get("id")
    if not lead_id:
        logger.error("lead_missing_id")
        return

    name   = lead.get("name") or "Unknown"
    source = lead.get("source") or "—"
    course = lead.get("course") or "—"

    # ── Score + Classify ──────────────────────────────────────────────────────
    scoring_rules = _fetch_scoring_rules()
    final_score   = score_lead(lead, scoring_rules)
    lead_type     = classify_lead(final_score)
    action        = next_best_action(lead_type, lead)
    ltv           = predict_ltv(final_score, lead)

    # Single summary log — everything a counsellor or engineer needs at a glance
    type_icon = {"Hot": "🔥", "Warm": "🌡", "Cold": "❄"}.get(lead_type, "")
    logger.info(
        f"{type_icon} {name} | {source} → {lead_type} ({final_score}) | ₹{ltv:,.0f} | {action}",
    )

    # ── Write payload ─────────────────────────────────────────────────────────
    payload = {
        "aiScore":        final_score,
        "score":          final_score,
        "type":           lead_type,
        "nextBestAction": action,
        "predictedLTV":   ltv,
    }
    await _write_scored_payload(lead_id, payload)

    # ── Side effects ──────────────────────────────────────────────────────────
    _log_ai_activity(lead_id, final_score, lead_type, action, ltv)
    _send_notification(lead, lead_type, action)


def _fetch_scoring_rules() -> list[dict]:
    try:
        return (
            supabase.table("ScoringRule")
            .select("*")
            .eq("active", True)
            .execute()
            .data
        ) or []
    except Exception as exc:
        # Demoted to debug — engine falls back to signal scoring automatically.
        # If this keeps failing, the StreamReset fix below resolves it.
        logger.debug("scoring_rules_unavailable", error=str(exc))
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  POSTGRES LISTEN / NOTIFY LOOP
# ══════════════════════════════════════════════════════════════════════════════

def _make_pg_connection() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(settings.DATABASE_URL)
    conn.set_isolation_level(0)           # AUTOCOMMIT — mandatory for LISTEN
    conn.cursor().execute("LISTEN new_lead;")
    return conn


async def listen_loop():
    loop = asyncio.get_event_loop()

    logger.info("pg_connecting")
    conn = await loop.run_in_executor(None, _make_pg_connection)
    logger.info("pg_listening", channel="new_lead")

    while True:
        try:
            # select() with 5s timeout so the loop stays async-friendly
            await loop.run_in_executor(None, lambda: select.select([conn], [], [], 5))
            conn.poll()

            while conn.notifies:
                notify = conn.notifies.pop(0)
                try:
                    lead = json.loads(notify.payload)
                    # Fire-and-forget per lead — don't block the listener
                    asyncio.create_task(process_lead(lead))
                except json.JSONDecodeError as exc:
                    logger.error("payload_parse_error", error=str(exc), raw=notify.payload[:200])

        except psycopg2.OperationalError as exc:
            logger.error("pg_connection_lost", error=str(exc))
            await asyncio.sleep(5)
            try:
                conn = await loop.run_in_executor(None, _make_pg_connection)
                logger.info("pg_reconnected")
            except Exception as reconnect_exc:
                logger.critical("pg_reconnect_failed", error=str(reconnect_exc))
                raise


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    logger.info(f"F2Fintech Lead Agent starting → {settings.CRM_API_BASE}")

    # CRM reachability check — log only on failure
    async with httpx.AsyncClient(timeout=10) as http:
        try:
            r = await http.get(settings.CRM_API_BASE)
            if r.status_code >= 500:
                logger.warning("crm_unhealthy", status=r.status_code)
            # 404 is fine — base URL isn't a real endpoint, just a reachability ping
        except Exception as exc:
            logger.warning(f"CRM unreachable at startup — will retry per-lead | {exc}")

    # Supabase reachability — log only on failure
    try:
        supabase.table("Lead").select("id").limit(1).execute()
    except Exception as exc:
        logger.error(f"Supabase unreachable — StreamReset usually means HTTP/2 issue | {exc}")
        logger.error("Fix: set HTTP2=false in your httpx client or upgrade supabase-py")

    asyncio.create_task(_dlq_reprocessor())
    await listen_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("agent_stopped_by_user")
    except Exception as exc:
        logger.critical("agent_fatal", error=str(exc))
        raise