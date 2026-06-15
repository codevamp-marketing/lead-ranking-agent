"""
agent/welcome_service.py — Welcome Message Service
====================================================
Sends an instant welcome message to every new lead via:
  • WhatsApp  (Twilio — sandbox OR production)
  • SMS       (DISABLED — commented out, kept for reference)
  • Email     (SendGrid — optional, not yet wired)

DEMO SETUP (current):
  WHATSAPP_MODE=sandbox → clean welcome goes to ADMIN_PHONE
  Set ADMIN_PHONE in .env — do NOT hardcode it here.

FIXES IN THIS VERSION
─────────────────────
  FIX 1 — ADMIN_PHONE read from settings/.env, never hardcoded.

  FIX 2 — Per-phone dedup guard prevents double-send for duplicate leads.

  FIX 3 — SMS COMPLETELY DISABLED.
           _send_sms() and _send_admin_lead_alert() are kept for
           reference but are NEVER called from anywhere. To re-enable,
           uncomment the relevant line in _send_phone_message().

  FIX 4 — SANDBOX SENDS CLEAN WELCOME ONLY (root cause of double-message bug).
           Root cause: _send_whatsapp_sandbox() was sending admin_notification
           which prepended "[DEMO] New Lead!\\nName: ...\\nPhone: ...\\n" before
           the welcome body. This caused two distinct message blocks to arrive:
           one that looked like an SMS alert, one that was the actual welcome.
           Fix: sandbox now sends exactly `body` (the clean welcome text),
           identical to what production sends. The [DEMO] block is gone entirely.
           If you need lead details on new leads, use the /health endpoint or
           check Supabase directly — don't contaminate the WhatsApp channel.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Optional

from utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  ADMIN CONTACT — read from settings/.env, never hardcoded
# ═══════════════════════════════════════════════════════════════════════════════

def _get_admin_phone() -> str:
    """Returns ADMIN_PHONE from settings (populated from .env)."""
    phone = getattr(settings, "ADMIN_PHONE", "").strip()
    if not phone:
        logger.warning(
            "admin_phone_not_in_settings",
            fix="Add ADMIN_PHONE=+91XXXXXXXXXX to .env and ADMIN_PHONE field to settings.py",
        )
        phone = "+919634776903"   # last-resort fallback only
    return phone


ADMIN_EMAIL = "mailmekhan76@gmail.com"


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-PHONE DEDUP GUARD
# ═══════════════════════════════════════════════════════════════════════════════

_RECENT_WELCOME_SENDS: dict[str, float] = {}
WELCOME_DEDUP_WINDOW_SECS = 10   # block second send to same phone within 10s


def _should_send_welcome(phone: str) -> bool:
    now  = time.monotonic()
    last = _RECENT_WELCOME_SENDS.get(phone, 0.0)
    if now - last < WELCOME_DEDUP_WINDOW_SECS:
        logger.warning(
            "welcome_dedup_blocked",
            phone=phone,
            seconds_since_last=round(now - last, 1),
            hint="Duplicate lead in DB — fix Lead table to prevent this",
        )
        return False
    _RECENT_WELCOME_SENDS[phone] = now

    cutoff = now - WELCOME_DEDUP_WINDOW_SECS * 2
    stale  = [p for p, t in _RECENT_WELCOME_SENDS.items() if t < cutoff]
    for p in stale:
        del _RECENT_WELCOME_SENDS[p]

    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  MESSAGE TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

WELCOME_WHATSAPP_TEXT = (
    "Hello {name}! 👋\n\n"
    "Thank you for your interest in Invertis University!\n"
    "We have received your admission inquiry and our counsellor "
    "will contact you shortly.\n\n"
    "You can also ask me about:\n"
    "• Courses & fees\n"
    "• Eligibility criteria\n"
    "• Hostel & campus\n"
    "• Admission process\n\n"
    "— Admissions Team, Invertis University"
)

# Kept for reference only — SMS is disabled
ADMIN_SMS_TEMPLATE = (
    "New Lead!\n"
    "Name:   {name}\n"
    "Phone:  {phone}\n"
    "Email:  {email}\n"
    "Course: {course}\n"
    "Source: {source}"
)

WELCOME_EMAIL_SUBJECT = "Thank you for your interest — Invertis University"
WELCOME_EMAIL_HTML = """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: auto;">
  <h2 style="color: #2c3e50;">Hello {name}!</h2>
  <p>Thank you for your interest in <strong>Invertis University</strong>.</p>
  <p>We have received your admission inquiry and our counsellor
     will contact you shortly.</p>
  <p>If you have any immediate questions, feel free to reply to this email.</p>
  <br/>
  <p style="color: #7f8c8d;">— Admissions Team, Invertis University</p>
</div>
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  STARTUP VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

def validate_twilio_config() -> bool:
    mode        = (settings.WHATSAPP_MODE or "disabled").lower().strip()
    admin_phone = _get_admin_phone()

    if mode == "disabled":
        logger.info("twilio_disabled")
        return True

    missing = []
    if not settings.TWILIO_ACCOUNT_SID:
        missing.append("TWILIO_ACCOUNT_SID")
    if not settings.TWILIO_AUTH_TOKEN:
        missing.append("TWILIO_AUTH_TOKEN")
    if mode == "sms" and not settings.TWILIO_SMS_FROM:
        missing.append("TWILIO_SMS_FROM")
    if mode in ("sandbox", "production") and not settings.TWILIO_WHATSAPP_FROM:
        missing.append("TWILIO_WHATSAPP_FROM")

    if missing:
        logger.error("twilio_config_missing", missing=missing)
        return False

    if not settings.TWILIO_ACCOUNT_SID.startswith("AC"):
        logger.error("twilio_invalid_sid")
        return False

    sanitised = _sanitise_phone(admin_phone)
    if not sanitised:
        logger.error(
            "admin_phone_invalid",
            admin_phone=admin_phone,
            fix="ADMIN_PHONE in .env must be E.164 format e.g. +919634776903",
        )
        return False

    logger.info(
        "twilio_config_ok",
        mode        = mode,
        admin_phone = admin_phone,
        from_number = settings.TWILIO_WHATSAPP_FROM,
    )
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  PHONE SANITISER
# ═══════════════════════════════════════════════════════════════════════════════

def _sanitise_phone(raw: str) -> Optional[str]:
    """Converts any Indian phone format to E.164 (+91XXXXXXXXXX)."""
    digits = re.sub(r"[^\d]", "", raw)
    if digits.startswith("91") and len(digits) == 12:
        return f"+{digits}"
    if len(digits) == 10:
        return f"+91{digits}"
    if digits.startswith("0") and len(digits) == 11:
        return f"+91{digits[1:]}"
    logger.warning("phone_invalid", raw=raw, digits=len(digits))
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  TWILIO CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

def _get_twilio_client():
    try:
        from twilio.rest import Client
        return Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    except ImportError:
        raise RuntimeError("twilio not installed — pip install twilio")


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVERSATION STORE — deferred import prevents circular import
# ═══════════════════════════════════════════════════════════════════════════════

async def _store_outbound(
    lead_phone: str, lead_name: str,
    body: str, channel: str, sid: str,
    message_type: str = "welcome",
) -> None:
    try:
        from webhook.conversation_store import record_message
        await record_message(
            lead_phone=lead_phone, lead_name=lead_name,
            direction="outbound", body=body, channel=channel, sid=sid,
            message_type=message_type,
        )
    except Exception as exc:
        logger.warning("store_outbound_failed", error=str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
#  ADMIN LEAD ALERT — DISABLED
#  Kept for reference. Re-enable by uncommenting in send_welcome_messages().
# ═══════════════════════════════════════════════════════════════════════════════

async def _send_admin_lead_alert(lead: dict) -> None:
    """SMS lead alert — DISABLED. Uncomment call in send_welcome_messages() to enable."""
    if not settings.TWILIO_SMS_FROM:
        return

    admin_phone = _get_admin_phone()
    name   = lead.get("name")   or "N/A"
    phone  = lead.get("phone")  or "N/A"
    course = lead.get("course") or "N/A"
    source = lead.get("source") or "N/A"
    score  = lead.get("aiScore") or "—"

    body = (
        f"[NEW LEAD ALERT]\n"
        f"Name:     {name}\n"
        f"Phone:    {phone}\n"
        f"Course:   {course}\n"
        f"Source:   {source}\n"
        f"AI Score: {score}"
    )

    try:
        def _send():
            client = _get_twilio_client()
            return client.messages.create(
                body=body, from_=settings.TWILIO_SMS_FROM, to=admin_phone,
            ).sid
        sid = await asyncio.to_thread(_send)
        logger.info("admin_alert_sent", to=admin_phone, sid=sid)
    except Exception as exc:
        logger.warning("admin_alert_failed", error=str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
#  SMS WELCOME — DISABLED
#  Kept for reference. Re-enable by uncommenting in _send_phone_message().
# ═══════════════════════════════════════════════════════════════════════════════

async def _send_sms(lead: dict) -> None:
    """SMS welcome — DISABLED. Uncomment in _send_phone_message() to re-enable."""
    if not all([settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN, settings.TWILIO_SMS_FROM]):
        logger.warning("sms_skipped_no_credentials")
        return

    admin_phone = _get_admin_phone()
    name   = (lead.get("name")   or "N/A").strip()
    phone  = (lead.get("phone")  or "N/A").strip()
    email  = (lead.get("email")  or "N/A").strip()
    course = (lead.get("course") or "N/A").strip()
    source = (lead.get("source") or "N/A").strip()

    body = ADMIN_SMS_TEMPLATE.format(
        name=name, phone=phone, email=email, course=course, source=source,
    )

    try:
        def _send():
            client = _get_twilio_client()
            return client.messages.create(
                body=body, from_=settings.TWILIO_SMS_FROM, to=admin_phone,
            ).sid

        sid = await asyncio.to_thread(_send)
        logger.info("sms_sent", to=admin_phone, sid=sid)
        await _store_outbound(admin_phone, name, body, "sms", sid)

    except Exception as exc:
        err = str(exc)
        if "21608" in err or "unverified" in err.lower():
            logger.error("sms_unverified_number", fix=f"Verify {admin_phone} in Twilio console")
        else:
            logger.error("sms_failed", error=err)


# ═══════════════════════════════════════════════════════════════════════════════
#  WHATSAPP — PRODUCTION
# ═══════════════════════════════════════════════════════════════════════════════

async def _send_whatsapp_production(lead: dict) -> None:
    raw_phone = (lead.get("phone") or "").strip()
    if not raw_phone:
        logger.warning("prod_no_phone", lead_id=lead.get("id"))
        return

    phone = _sanitise_phone(raw_phone)
    if not phone:
        return

    name = lead.get("name") or "there"
    body = WELCOME_WHATSAPP_TEXT.format(name=name)

    try:
        def _send():
            client = _get_twilio_client()
            return client.messages.create(
                body=body, from_=settings.TWILIO_WHATSAPP_FROM,
                to=f"whatsapp:{phone}",
            ).sid

        sid = await asyncio.to_thread(_send)
        logger.info("whatsapp_prod_sent", to=phone, sid=sid)
        await _store_outbound(phone, name, body, "whatsapp", sid)

    except Exception as exc:
        logger.error("whatsapp_prod_failed", to=phone, error=str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
#  WHATSAPP — SANDBOX
#
#  FIX 4: Sends ONLY the clean welcome body — no [DEMO] operational block.
#
#  BEFORE (broken):
#    admin_notification = "[DEMO] New Lead!\nName: ...\nPhone: ...\n\n" + body
#    client.messages.create(body=admin_notification, ...)
#    This produced TWO visible blocks in WhatsApp — the metadata dump looked
#    like a separate SMS alert arriving before the welcome.
#
#  AFTER (fixed):
#    client.messages.create(body=body, ...)
#    Single clean welcome message, identical to production behaviour.
#    Lead details are available in Supabase / logs — not via WhatsApp.
# ═══════════════════════════════════════════════════════════════════════════════

async def _send_whatsapp_sandbox(lead: dict) -> None:
    admin_phone = _get_admin_phone()
    name        = lead.get("name") or "there"
    raw_phone   = (lead.get("phone") or "N/A").strip()

    admin = _sanitise_phone(admin_phone)
    if not admin:
        logger.error("admin_phone_invalid", phone=admin_phone)
        return

    # FIX 4: clean welcome body only — no [DEMO] prefix
    body = WELCOME_WHATSAPP_TEXT.format(name=name)

    try:
        def _send():
            client = _get_twilio_client()
            return client.messages.create(
                body  = body,   # ← FIX: was admin_notification (contained [DEMO] dump)
                from_ = settings.TWILIO_WHATSAPP_FROM,
                to    = f"whatsapp:{admin}",
            ).sid

        sid = await asyncio.to_thread(_send)
        logger.info(
            "whatsapp_sandbox_sent",
            sent_to    = admin,
            lead_phone = raw_phone,
            lead_name  = name,
            sid        = sid,
        )
        await _store_outbound(admin, name, body, "whatsapp", sid, message_type="welcome")

    except Exception as exc:
        logger.error("whatsapp_sandbox_failed", error=str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
#  CHANNEL ROUTER
#
#  FIX 3: SMS mode is a complete no-op. Only WhatsApp (sandbox/production) sends.
#  To re-enable SMS: uncomment `await _send_sms(lead)` below.
# ═══════════════════════════════════════════════════════════════════════════════

async def _send_phone_message(lead: dict) -> None:
    mode = (settings.WHATSAPP_MODE or "disabled").lower().strip()
    logger.info("sending_welcome", mode=mode, admin=_get_admin_phone())

    if mode == "sms":
        # SMS DISABLED — uncomment below to re-enable:
        # await _send_sms(lead)
        logger.info("sms_mode_disabled", hint="Set WHATSAPP_MODE=sandbox or production to use WhatsApp")
    elif mode == "production":
        await _send_whatsapp_production(lead)
    elif mode == "sandbox":
        await _send_whatsapp_sandbox(lead)
    elif mode == "disabled":
        logger.info("phone_disabled")
    else:
        logger.warning("unknown_mode", mode=mode)


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

async def send_welcome_messages(lead: dict) -> None:
    lead_phone = (lead.get("phone") or "").strip()
    canonical  = _sanitise_phone(lead_phone) or lead_phone

    if not _should_send_welcome(canonical):
        return

    await asyncio.gather(
        # _send_admin_lead_alert(lead),  # SMS alert — DISABLED
        _send_phone_message(lead),       # WhatsApp welcome only
        return_exceptions=True,
    )


async def send_reply(
    lead_phone: str,
    body:       str,
    lead_name:  str = "Lead",
    channel:    str = "whatsapp",
) -> Optional[str]:
    if not all([settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN]):
        logger.warning("send_reply_no_credentials")
        return None

    from_number = (
        settings.TWILIO_WHATSAPP_FROM if channel == "whatsapp"
        else settings.TWILIO_SMS_FROM
    )
    to_number = f"whatsapp:{lead_phone}" if channel == "whatsapp" else lead_phone

    if not from_number:
        logger.warning("send_reply_no_from_number")
        return None

    try:
        def _send():
            client = _get_twilio_client()
            return client.messages.create(
                body=body, from_=from_number, to=to_number,
            ).sid

        sid = await asyncio.to_thread(_send)
        logger.info("reply_sent", to=lead_phone, sid=sid)
        await _store_outbound(lead_phone, lead_name, body, channel, sid, message_type=None)
        return sid

    except Exception as exc:
        logger.error("reply_failed", to=lead_phone, error=str(exc))
        return None