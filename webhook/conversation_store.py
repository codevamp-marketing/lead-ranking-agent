"""
webhook/conversation_store.py — Conversation Persistence Layer
==============================================================
Stores and retrieves all inbound/outbound WhatsApp/SMS messages.

Root cause fixes in this version
─────────────────────────────────

FIX 1 — SyncQueryRequestBuilder has no attribute 'select'
  supabase-py 2.28 does NOT support .upsert().select() chaining.
  The upsert returns a SyncQueryRequestBuilder, not a SyncSelectRequestBuilder,
  so .select() doesn't exist on it.

  Wrong (crashes):
    db.table("Conversation").upsert({...}).select("id").execute()

  Fix: split into two operations:
    Step A — upsert (no select chain)
    Step B — select to get the UUID back

FIX 2 — StreamReset stream_id:1, error_code:1
  supabase-py 2.x ships with httpx HTTP/2 enabled by default.
  Supabase's pooler resets HTTP/2 streams.
  Fix: patch the postgrest session to use httpx.Client(http2=False).

FIX 3 — Schema alignment
  Previous version wrote to "conversations" (flat, lowercase, doesn't exist).
  This version writes to the tables you actually created:
    "Conversation"        (one row per lead phone)
    "ConversationMessage" (one row per message)
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Optional

import httpx
from supabase import create_client, Client
from utils.logger import get_logger

logger = get_logger(__name__)

# ── ANSI colours ──────────────────────────────────────────────────────────────
_RESET = "\033[0m"
_GREEN = "\033[92m"
_CYAN  = "\033[96m"
_BOLD  = "\033[1m"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ═══════════════════════════════════════════════════════════════════════════════
#  SUPABASE CLIENT — HTTP/1.1 forced (StreamReset fix)
# ═══════════════════════════════════════════════════════════════════════════════

_supabase: Optional[Client] = None


def _get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        url = os.environ.get("SUPABASE_URL", "").strip().strip('"').strip("'")
        key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip().strip('"').strip("'")

        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")

        _supabase = create_client(url, key)

        # FIX 2: Force HTTP/1.1 — prevents StreamReset from Supabase pooler
        try:
            _supabase.postgrest.session = httpx.Client(http2=False, timeout=15)
        except Exception:
            pass  # older supabase-py — best effort

    return _supabase


# ═══════════════════════════════════════════════════════════════════════════════
#  TERMINAL PRINT
# ═══════════════════════════════════════════════════════════════════════════════

def _print_message(
    direction: str, lead_phone: str, lead_name: str,
    body: str, channel: str, sid: Optional[str] = None,
) -> None:
    colour   = _GREEN if direction == "inbound" else _CYAN
    arrow    = "↓ INBOUND " if direction == "inbound" else "↑ OUTBOUND"
    sid_line = f"  SID : {sid}\n" if sid else ""
    print(
        f"\n{colour}{_BOLD}{'─' * 60}\n"
        f"  {arrow}  [{channel.upper()}]  {_now_iso()}\n"
        f"{'─' * 60}{_RESET}\n"
        f"{colour}  Lead  : {lead_name} ({lead_phone})\n"
        f"  Msg   : {body[:200]}\n{sid_line}{_RESET}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  IN-MEMORY FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

_memory_store: dict[str, list[dict]] = {}


def _memory_append(
    lead_phone: str, direction: str, body: str,
    channel: str, sid: Optional[str],
) -> None:
    if lead_phone not in _memory_store:
        _memory_store[lead_phone] = []
    _memory_store[lead_phone].append({
        "direction": direction, "body": body,
        "channel": channel, "sid": sid, "timestamp": _now_iso(),
    })


def get_conversation_memory(lead_phone: str) -> list[dict]:
    return _memory_store.get(lead_phone, [])


# ═══════════════════════════════════════════════════════════════════════════════
#  SUPABASE WRITES — correct API for supabase-py 2.28
# ═══════════════════════════════════════════════════════════════════════════════

def _upsert_conversation_sync(lead_phone: str, lead_name: str) -> Optional[str]:
    """
    Finds or creates a Conversation row for this lead phone.
    Returns the UUID of the row, or None on failure.

    FIX 1: supabase-py 2.28 does NOT support .upsert().select() chaining.
    Solution: two-step approach:
      Step A — upsert the row (no .select() chain)
      Step B — immediately SELECT to get the UUID back

    This is the correct pattern for supabase-py 2.x.
    """
    db = _get_supabase()

    try:
        # Step A — upsert (insert or update on conflict)
        # Do NOT chain .select() here — it crashes in 2.28
        db.table("Conversation").upsert(
            {
                "leadId":    lead_phone,
                "leadName":  lead_name or "Unknown",
                "updatedAt": _now_iso(),
            },
            on_conflict="leadId",
        ).execute()

    except Exception as exc:
        logger.error("conversation_upsert_failed", lead_phone=lead_phone, error=str(exc))
        return None

    try:
        # Step B — fetch the UUID (separate query — always works in 2.28)
        result = (
            db.table("Conversation")
            .select("id")
            .eq("leadId", lead_phone)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        if rows:
            return rows[0]["id"]

        logger.error(
            "conversation_row_not_found_after_upsert",
            lead_phone=lead_phone,
            hint="Row was not created — check RLS policies on Conversation table in Supabase",
        )
        return None

    except Exception as exc:
        logger.error("conversation_select_failed", lead_phone=lead_phone, error=str(exc))
        return None


def _insert_message_sync(
    conversation_id: str, direction: str,
    body: str, channel: str, sid: Optional[str],
) -> bool:
    """
    Inserts one ConversationMessage row.
    Returns True on success, False on failure.
    """
    try:
        db = _get_supabase()
        db.table("ConversationMessage").insert({
            "conversationId": conversation_id,
            "direction":      direction,
            "body":           body or "",
            "channel":        channel,
            "twilioSid":      sid,
            "createdAt":      _now_iso(),
        }).execute()
        return True
    except Exception as exc:
        logger.error(
            "message_insert_failed",
            conversation_id=conversation_id,
            direction=direction,
            error=str(exc),
        )
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

async def record_message(
    lead_phone: str,
    lead_name:  str,
    direction:  str,              # "inbound" | "outbound"
    body:       str,
    channel:    str = "whatsapp",
    sid:        Optional[str] = None,
) -> bool:
    """
    Records a message to terminal + in-memory + Supabase.
    Never raises. Returns True if Supabase write succeeded.
    """
    # 1. Terminal — always instant
    _print_message(direction, lead_phone, lead_name, body, channel, sid)

    # 2. In-memory fallback — always
    _memory_append(lead_phone, direction, body, channel, sid)

    # 3. Supabase — sync calls in thread pool (non-blocking to event loop)
    conversation_id = await asyncio.to_thread(
        _upsert_conversation_sync, lead_phone, lead_name
    )

    if not conversation_id:
        logger.warning(
            "message_stored_in_memory_only",
            lead_phone=lead_phone,
            direction=direction,
            hint=(
                "Check: (1) conversations.sql was run in Supabase SQL Editor, "
                "(2) RLS is disabled on Conversation table or service key is used"
            ),
        )
        return False

    success = await asyncio.to_thread(
        _insert_message_sync, conversation_id, direction, body, channel, sid
    )

    if success:
        logger.info("message_saved", direction=direction, lead_phone=lead_phone, channel=channel)

    return success


async def get_conversation_history(lead_phone: str, limit: int = 50) -> list[dict]:
    """
    Fetches conversation history from Supabase, oldest-first.
    Falls back to in-memory if Supabase is unavailable.
    """
    def _fetch() -> list[dict]:
        db = _get_supabase()

        conv = (
            db.table("Conversation")
            .select("id")
            .eq("leadId", lead_phone)
            .limit(1)
            .execute()
        )
        rows = conv.data or []
        if not rows:
            return []

        conv_id = rows[0]["id"]
        msgs = (
            db.table("ConversationMessage")
            .select("direction, body, channel, twilioSid, createdAt")
            .eq("conversationId", conv_id)
            .order("createdAt", desc=False)
            .limit(limit)
            .execute()
        )
        return msgs.data or []

    try:
        return await asyncio.to_thread(_fetch)
    except Exception as exc:
        logger.warning("get_history_failed", lead_phone=lead_phone, error=str(exc))
        return get_conversation_memory(lead_phone)