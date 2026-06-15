"""
webhook/conversation_store.py — Conversation Store v6.0
=========================================================
FIXES v6.0
──────────
FIX 1 — _get_supabase() now reads os.environ AFTER load_dotenv() has run.
  Root cause: supabase client was being built with empty URL because
  config.settings was imported before .env was loaded. Now we call
  load_dotenv(override=True) inside _get_supabase() as a safety net,
  and read directly from os.environ — NOT through config.settings.

FIX 2 — Removed postgrest.session override. supabase-py >= 2.x doesn't
  support that attribute, and setting it caused UnsupportedProtocol errors
  by replacing the internal httpx client with a misconfigured one.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv(override=True)

import asyncio
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

_RESET = "\033[0m"
_GREEN = "\033[92m"
_CYAN  = "\033[96m"
_BOLD  = "\033[1m"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalise_phone(raw: str) -> str:
    if not raw:
        return ""
    cleaned = raw.strip().replace("whatsapp:", "").strip()
    digits  = re.sub(r"[^\d]", "", cleaned)
    if digits.startswith("91") and len(digits) == 12:
        return f"+{digits}"
    if len(digits) == 10:
        return f"+91{digits}"
    if digits.startswith("0") and len(digits) == 11:
        return f"+91{digits[1:]}"
    return cleaned if cleaned.startswith("+") else f"+{digits}" if digits else cleaned


def phone_variants(phone: str) -> list[str]:
    e164   = normalise_phone(phone)
    digits = re.sub(r"[^\d]", "", e164)
    ten    = digits[-10:] if len(digits) >= 10 else digits
    return list({e164, digits, ten, f"+91{ten}"})


# ── In-memory message store ───────────────────────────────────────────────────
_memory_store: dict[str, list[dict]] = {}
_MAX_MEMORY_PER_PHONE = 100


def _memory_append(phone: str, direction: str, body: str, channel: str,
                   sid: Optional[str], message_type: Optional[str] = None) -> None:
    key = normalise_phone(phone)
    if key not in _memory_store:
        _memory_store[key] = []
    _memory_store[key].append({
        "direction":   direction,
        "body":        body,
        "channel":     channel,
        "twilioSid":   sid,
        "messageType": message_type,
        "createdAt":   _now_iso(),
    })
    if len(_memory_store[key]) > _MAX_MEMORY_PER_PHONE:
        _memory_store[key] = _memory_store[key][-_MAX_MEMORY_PER_PHONE:]


def get_conversation_memory(phone: str) -> list[dict]:
    key = normalise_phone(phone)
    return list(_memory_store.get(key, []))


# ── Supabase client — lazy, always reads from os.environ ─────────────────────
_supabase: Optional[object] = None


def _get_supabase():
    global _supabase
    if _supabase is None:
        # Always call load_dotenv here as a safety net
        load_dotenv(override=True)
        url = os.environ.get("SUPABASE_URL", "").strip().strip('"').strip("'")
        key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip().strip('"').strip("'")
        if not url or not url.startswith("http"):
            raise RuntimeError(f"SUPABASE_URL invalid: '{url}'")
        if not key:
            raise RuntimeError("SUPABASE_SERVICE_KEY missing")
        from supabase import create_client
        _supabase = create_client(url, key)
        logger.info("conversation_store_supabase_ok")
    return _supabase


def get_supabase_client():
    return _get_supabase()


# ── Console logging ───────────────────────────────────────────────────────────

def _print_message(direction: str, lead_phone: str, lead_name: str,
                   body: str, channel: str, sid: Optional[str] = None) -> None:
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


# ── Lead resolution ───────────────────────────────────────────────────────────

def _resolve_lead_sync(lead_phone: str) -> Optional[str]:
    try:
        db       = _get_supabase()
        variants = phone_variants(lead_phone)
        result   = db.table("Lead").select("id").in_("phone", variants).order("createdAt", desc=True).limit(1).execute()
        rows     = result.data or []
        return rows[0]["id"] if rows else None
    except Exception as exc:
        logger.warning("lead_resolution_failed", phone=lead_phone, error=str(exc))
        return None


def _insert_message_sync(lead_id: str, lead_phone: str, direction: str,
                         body: str, channel: str, sid: Optional[str],
                         message_type: Optional[str] = None) -> bool:
    try:
        db      = _get_supabase()
        payload = {
            "leadId":    lead_id,
            "phone":     normalise_phone(lead_phone),
            "direction": direction,
            "body":      body or "",
            "channel":   channel,
            "twilioSid": sid,
            "createdAt": _now_iso(),
        }
        if message_type:
            payload["messageType"] = message_type
        db.table("Message").insert(payload).execute()
        return True
    except Exception as exc:
        logger.warning("message_insert_failed", lead_id=lead_id, error=str(exc))
        return False


# ── Public API ────────────────────────────────────────────────────────────────

async def record_message(
    lead_phone:   str,
    lead_name:    str,
    direction:    str,
    body:         str,
    channel:      str          = "whatsapp",
    sid:          Optional[str] = None,
    message_type: Optional[str] = None,
) -> bool:
    norm_phone = normalise_phone(lead_phone)
    _print_message(direction, norm_phone, lead_name, body, channel, sid)
    _memory_append(norm_phone, direction, body, channel, sid, message_type)
    try:
        lead_id = await asyncio.to_thread(_resolve_lead_sync, norm_phone)
        if not lead_id:
            logger.debug("message_memory_only_no_lead", phone=norm_phone)
            return False
        success = await asyncio.to_thread(
            _insert_message_sync, lead_id, norm_phone, direction, body, channel, sid, message_type
        )
        if success:
            logger.debug("message_saved_db", direction=direction, phone=norm_phone)
        return success
    except Exception as exc:
        logger.warning("record_message_db_failed", phone=norm_phone, error=str(exc))
        return False


_EXCLUDED_TYPES = {"welcome", "system_notification"}
_EXCLUDED_BODY_PREFIXES = ("[DEMO]", "New Lead!", "[NEW LEAD ALERT]")


def _is_conversation_message(msg: dict) -> bool:
    msg_type = msg.get("messageType") or msg.get("message_type") or ""
    if msg_type in _EXCLUDED_TYPES:
        return False
    body = msg.get("body", "")
    if any(body.startswith(p) for p in _EXCLUDED_BODY_PREFIXES):
        return False
    if msg.get("direction") == "outbound" and len(body) > 1000:
        return False
    return True


async def get_conversation_history(lead_phone: str, limit: int = 20) -> list[dict]:
    norm_phone = normalise_phone(lead_phone)
    variants   = phone_variants(lead_phone)
    try:
        def _fetch() -> list[dict]:
            db     = _get_supabase()
            result = db.table("Message").select(
                "direction, body, channel, twilioSid, createdAt, messageType"
            ).in_("phone", variants).order("createdAt", desc=True).limit(limit * 3).execute()
            rows     = result.data or []
            filtered = [r for r in rows if _is_conversation_message(r)]
            filtered.reverse()
            return filtered[-limit:]

        db_history = await asyncio.to_thread(_fetch)
        if db_history:
            logger.debug("history_from_db", phone=norm_phone, count=len(db_history))
            return db_history
    except Exception as exc:
        logger.warning("history_db_failed", phone=norm_phone, error=str(exc))

    mem          = get_conversation_memory(norm_phone)
    filtered_mem = [m for m in mem if _is_conversation_message(m)]
    logger.debug("history_from_memory", phone=norm_phone, count=len(filtered_mem))
    return filtered_mem[-limit:]


async def _lookup_lead_by_phone(phone: str) -> Optional[dict]:
    try:
        db       = _get_supabase()
        variants = phone_variants(phone)
        result   = await asyncio.to_thread(
            lambda: db.table("Lead").select(
                "id, name, phone, email, pickedBy, aiScore, type"
            ).in_("phone", variants).order("createdAt", desc=True).limit(1).execute()
        )
        rows = result.data or []
        return rows[0] if rows else None
    except Exception as exc:
        logger.warning("lead_lookup_failed", phone=phone, error=str(exc))
        return None