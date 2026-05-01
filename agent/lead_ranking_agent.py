"""
agent/lead_ranking_agent.py — Invertis Lead Ranking Agent (Production)
========================================================================
Listens for new Lead inserts via Postgres LISTEN/NOTIFY, scores each
lead using a two-layer engine, writes AI fields back to the CRM, and
sends a welcome message.

Entry point is main.py — this file contains only business logic.

FIXES (v2.2)
────────────
  • _get_supabase(): HTTP/1.1 forced via httpx.Client(http2=False) to
    eliminate StreamReset (error_code:1) from Supabase pooler.
    Same fix as conversation_store.py — both clients must use HTTP/1.1.
  • All other logic unchanged from v2.1.
"""

from __future__ import annotations

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

from agent.welcome_service import send_welcome_messages
from scoring.engine import score_lead, classify_lead, predict_ltv, next_best_action
from utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  SUPABASE CLIENT — HTTP/1.1 forced (StreamReset fix)
# ═══════════════════════════════════════════════════════════════════════════════

supabase: Optional[Client] = None


def _get_supabase() -> Client:
    """
    Lazy singleton Supabase client.

    StreamReset fix: after creating the client, patch the postgrest session
    to use HTTP/1.1. supabase-py 2.x uses httpx with HTTP/2 by default,
    which Supabase's pooler resets (StreamReset stream_id:1, error_code:1).

    httpx.Client(http2=False) forces HTTP/1.1 — reliable with Supabase pooler.
    """
    global supabase
    if supabase is None:
        try:
            from supabase.lib.client_options import ClientOptions
            supabase = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_SERVICE_KEY,
                options=ClientOptions(
                    postgrest_client_timeout=15,
                    storage_client_timeout=15,
                ),
            )
        except Exception:
            supabase = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_SERVICE_KEY,
            )

        # Apply HTTP/1.1 fix to this client's postgrest session
        try:
            supabase.postgrest.session = httpx.Client(http2=False, timeout=15)
        except Exception:
            pass  # older supabase-py — continue without fix

    return supabase


# ═══════════════════════════════════════════════════════════════════════════════
#  DEAD-LETTER QUEUE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DLQEntry:
    lead_id:   str
    payload:   dict
    attempts:  int   = 0
    queued_at: float = field(default_factory=time.time)

_dlq: deque[DLQEntry] = deque(maxlen=1000)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDE-EFFECT WRITERS
# ═══════════════════════════════════════════════════════════════════════════════

import uuid

async def _log_ai_activity(lead_id: str, score: int, lead_type: str, action: str, ltv: float) -> None:
    description = (
        f"AI ranked this lead as '{lead_type}' (score {score}/100). "
        f"Predicted LTV: ₹{ltv:,.0f}. Recommended action: {action}."
    )
    try:
        db = _get_supabase()
        await asyncio.to_thread(
            lambda: db.table("Activity").insert({
                "id":          str(uuid.uuid4()),
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
        )
    except Exception as exc:
        logger.warning("activity_log_failed", lead_id=lead_id, error=str(exc))


async def _send_notification(lead: dict, lead_type: str, action: str) -> None:
    counsellor_id = lead.get("pickedBy") or lead.get("createdBy")
    if not counsellor_id:
        return

    lead_id   = lead.get("id")
    lead_name = lead.get("name") or "A new lead"
    title_map = {
        "Hot":  f"🔥 Hot Lead: {lead_name} — Call Now",
        "Warm": f"🌡 Warm Lead: {lead_name} — Follow Up",
        "Cold": f"❄ Cold Lead: {lead_name} — Nurture",
    }

    try:
        db = _get_supabase()
        await asyncio.to_thread(
            lambda: db.table("notifications").insert({
                "id":     str(uuid.uuid4()),
                "userId": counsellor_id,
                "leadId": lead_id,
                "title":  title_map.get(lead_type, f"New Lead: {lead_name}"),
                "body":   f"AI Score: {lead.get('aiScore', '—')} | Action: {action}",
                "read":   False,
            }).execute()
        )
        logger.info("notification_sent", lead_id=lead_id, counsellor_id=counsellor_id)
    except Exception as exc:
        logger.warning("notification_failed", lead_id=lead_id, error=str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
#  CRM PATCH — exponential backoff + jitter
# ═══════════════════════════════════════════════════════════════════════════════

_MAX_ATTEMPTS   = 3
_BASE_DELAY_SEC = 2.0
_MAX_DELAY_SEC  = 16.0


async def _patch_crm(lead_id: str, payload: dict) -> bool:
    if not settings.CRM_API_BASE:
        logger.debug("crm_skipped_no_base_url", lead_id=lead_id)
        return False

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
            ceiling = min(_MAX_DELAY_SEC, _BASE_DELAY_SEC * (2 ** (attempt - 1)))
            delay   = random.uniform(0, ceiling)
            logger.info("crm_retry_backoff", lead_id=lead_id, attempt=attempt, wait_sec=round(delay, 2))
            await asyncio.sleep(delay)

    logger.error("crm_all_retries_exhausted", lead_id=lead_id)
    return False


async def _write_scored_payload(lead_id: str, payload: dict) -> bool:
    if await _patch_crm(lead_id, payload):
        return True

    logger.warning("crm_exhausted_trying_supabase_direct", lead_id=lead_id)
    try:
        db = _get_supabase()
        await asyncio.to_thread(
            lambda: db.table("Lead").update(payload).eq("id", lead_id).execute()
        )
        logger.info("supabase_direct_write_ok", lead_id=lead_id)
        return True
    except Exception as exc:
        logger.error("supabase_direct_write_failed", lead_id=lead_id, error=str(exc))

    _dlq.append(DLQEntry(lead_id=lead_id, payload=payload))
    logger.error("lead_pushed_to_dlq", lead_id=lead_id, dlq_size=len(_dlq))
    return False


async def _dlq_reprocessor() -> None:
    logger.info("dlq_reprocessor_started")
    while True:
        await asyncio.sleep(60)
        if not _dlq:
            continue

        logger.info("dlq_reprocessor_tick", pending=len(_dlq))
        batch = [_dlq.popleft() for _ in range(len(_dlq))]

        for entry in batch:
            entry.attempts += 1
            try:
                await asyncio.to_thread(
            lambda e=entry, d=_get_supabase(): d.table("Lead").update(e.payload).eq("id", e.lead_id).execute()
        )
                logger.info("dlq_reprocess_ok", lead_id=entry.lead_id, attempts=entry.attempts)
            except Exception as exc:
                logger.error("dlq_reprocess_failed", lead_id=entry.lead_id, attempts=entry.attempts, error=str(exc))
                if entry.attempts < 10:
                    _dlq.append(entry)
                else:
                    logger.critical(
                        "dlq_entry_abandoned",
                        lead_id=entry.lead_id, payload=entry.payload, queued_at=entry.queued_at,
                    )


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE: PROCESS ONE LEAD
# ═══════════════════════════════════════════════════════════════════════════════

async def _fetch_scoring_rules() -> list[dict]:
    try:
        db = _get_supabase()
        result = await asyncio.to_thread(
            lambda: db.table("ScoringRule").select("*").eq("active", True).execute()
        )
        return result.data or []
    except Exception as exc:
        logger.debug("scoring_rules_unavailable", error=str(exc))
        return []


async def process_lead(lead: dict) -> None:
    lead_id = lead.get("id")
    if not lead_id:
        logger.error("lead_missing_id")
        return

    name   = lead.get("name") or "Unknown"
    source = lead.get("source") or "—"

    try:
        scoring_rules = await _fetch_scoring_rules()
        final_score   = score_lead(lead, scoring_rules)
        lead_type     = classify_lead(final_score)
        action        = next_best_action(lead_type, lead)
        ltv           = predict_ltv(final_score, lead)

        type_icon = {"Hot": "🔥", "Warm": "🌡", "Cold": "❄"}.get(lead_type, "")
        logger.info(f"{type_icon} {name} | {source} → {lead_type} ({final_score}) | ₹{ltv:,.0f} | {action}")

        payload = {
            "aiScore":        final_score,
            "score":          final_score,
            "type":           lead_type,
            "nextBestAction": action,
            "predictedLTV":   ltv,
        }
        lead.update(payload)

        await _write_scored_payload(lead_id, payload)
        await send_welcome_messages(lead)

        await asyncio.gather(
            _log_ai_activity(lead_id, final_score, lead_type, action, ltv),
            _send_notification(lead, lead_type, action),
            return_exceptions=True,
        )

    except Exception as exc:
        logger.error("process_lead_failed", lead_id=lead_id, name=name, error=str(exc), exc_info=True)


def _task_error_handler(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error("unhandled_task_exception", task=task.get_name(), error=str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
#  POSTGRES LISTEN / NOTIFY LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def _make_pg_connection() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(settings.DATABASE_URL)
    conn.set_isolation_level(0)
    conn.cursor().execute(f"LISTEN {settings.PG_NOTIFY_CHANNEL};")
    return conn


async def listen_loop() -> None:
    loop = asyncio.get_event_loop()

    logger.info("pg_connecting")
    conn = await loop.run_in_executor(None, _make_pg_connection)
    logger.info("pg_listening", channel=settings.PG_NOTIFY_CHANNEL)

    while True:
        try:
            await loop.run_in_executor(
                None,
                lambda: select.select([conn], [], [], settings.PG_SELECT_TIMEOUT),
            )
            conn.poll()

            while conn.notifies:
                notify = conn.notifies.pop(0)
                try:
                    lead = json.loads(notify.payload)
                except json.JSONDecodeError as exc:
                    logger.error("payload_parse_error", error=str(exc), raw=notify.payload[:200])
                    continue

                task = asyncio.create_task(process_lead(lead), name=f"lead-{lead.get('id', 'unknown')}")
                task.add_done_callback(_task_error_handler)

        except psycopg2.OperationalError as exc:
            logger.error("pg_connection_lost", error=str(exc))
            await asyncio.sleep(5)
            try:
                conn = await loop.run_in_executor(None, _make_pg_connection)
                logger.info("pg_reconnected")
            except Exception as reconnect_exc:
                logger.critical("pg_reconnect_failed", error=str(reconnect_exc))
                raise


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — called by main.py
# ═══════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    logger.info("agent_starting", crm=settings.CRM_API_BASE or "disabled")

    _get_supabase()  # init eagerly so StreamReset fix applied before first lead

    if settings.CRM_API_BASE:
        async with httpx.AsyncClient(timeout=10) as http:
            try:
                r = await http.get(settings.CRM_API_BASE)
                logger.info("crm_reachable", status=r.status_code) if r.status_code < 500 else logger.warning("crm_unhealthy", status=r.status_code)
            except Exception as exc:
                logger.warning("crm_unreachable_at_startup", error=str(exc))
    else:
        logger.info("crm_disabled_skipping_health_check")

    try:
        db = _get_supabase()
        await asyncio.to_thread(lambda: db.table("Lead").select("id").limit(1).execute())
        logger.info("supabase_reachable")
    except Exception as exc:
        logger.error("supabase_unreachable", error=str(exc))

    asyncio.create_task(_dlq_reprocessor(), name="dlq-reprocessor")
    await listen_loop()