"""
agent/welcome_service.py — Welcome Message Service
====================================================
Sends an instant welcome message to every new lead via:
  • SMS       (Twilio — demo mode: sends to ADMIN_PHONE only)
  • WhatsApp  (Twilio — sandbox OR production)
  • Email     (SendGrid — optional)

DEMO SETUP (current):
  WHATSAPP_MODE=sandbox → all messages go to ADMIN_PHONE (+918868058962)
  The lead's actual phone is shown in the message body for reference.

WHAT THE DEMO MESSAGE SHOWS:
  "[DEMO] New Lead!
   Name:   Shahbaz Khan
   Phone:  09634776903   ← this is the LEAD's phone (from DB), not ADMIN_PHONE
   ..."
  This is CORRECT behaviour. The message is SENT to ADMIN_PHONE (your number).
  The Phone: line just shows what phone the lead submitted in the form.

FIX in this version
─────────────────────
  • ADMIN_PHONE updated to +918868058962 (new demo number).
  • Added startup phone validation to catch formatting errors early.
  • _send_whatsapp_sandbox() now logs clearly which number it sends TO
    vs which phone the lead submitted, to avoid confusion.
"""

from __future__ import annotations

import asyncio
import re
from typing import Optional

from utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

# ── Admin contact (your demo WhatsApp number) ─────────────────────────────────
# This is WHERE messages are sent in sandbox mode.
# Must be joined to Twilio sandbox: text "join <keyword>" to +14155238886
ADMIN_PHONE = "+918868058962"
ADMIN_EMAIL = "mailmekhan76@gmail.com"

# ── Message templates ─────────────────────────────────────────────────────────

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
    mode = (settings.WHATSAPP_MODE or "disabled").lower().strip()
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

    # Validate ADMIN_PHONE format
    sanitised = _sanitise_phone(ADMIN_PHONE)
    if not sanitised:
        logger.error(
            "admin_phone_invalid",
            admin_phone=ADMIN_PHONE,
            fix="ADMIN_PHONE must be E.164 format e.g. +918868058962",
        )
        return False

    logger.info(
        "twilio_config_ok",
        mode       = mode,
        admin_phone= ADMIN_PHONE,
        from_number= settings.TWILIO_WHATSAPP_FROM,
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
) -> None:
    try:
        from webhook.conversation_store import record_message
        await record_message(
            lead_phone=lead_phone, lead_name=lead_name,
            direction="outbound", body=body, channel=channel, sid=sid,
        )
    except Exception as exc:
        logger.warning("store_outbound_failed", error=str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
#  SMS — DEMO MODE
# ═══════════════════════════════════════════════════════════════════════════════

async def _send_sms(lead: dict) -> None:
    if not all([settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN, settings.TWILIO_SMS_FROM]):
        logger.warning("sms_skipped_no_credentials")
        return

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
                body=body, from_=settings.TWILIO_SMS_FROM, to=ADMIN_PHONE,
            ).sid

        sid = await asyncio.to_thread(_send)
        logger.info("sms_sent", to=ADMIN_PHONE, sid=sid)
        await _store_outbound(ADMIN_PHONE, name, body, "sms", sid)

    except Exception as exc:
        err = str(exc)
        if "21608" in err or "unverified" in err.lower():
            logger.error("sms_unverified_number", fix=f"Verify {ADMIN_PHONE} in Twilio console")
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
# ═══════════════════════════════════════════════════════════════════════════════

async def _send_whatsapp_sandbox(lead: dict) -> None:
    """
    Sandbox mode: message is ALWAYS sent to ADMIN_PHONE (+918868058962).

    IMPORTANT: The lead's phone (e.g. 09634776903) shown in the message body
    is just informational — it's the phone number the lead submitted in the
    form. The actual Twilio destination is always ADMIN_PHONE in sandbox mode.

    To receive sandbox messages on a new number:
      1. Text "join <keyword>" to +14155238886 from the new number
      2. Wait for "You are all set" confirmation
      3. Update ADMIN_PHONE in this file to the new number
    """
    name      = lead.get("name") or "there"
    raw_phone = (lead.get("phone") or "N/A").strip()

    # Sanitise ADMIN_PHONE (the actual destination)
    admin = _sanitise_phone(ADMIN_PHONE)
    if not admin:
        logger.error("admin_phone_invalid", phone=ADMIN_PHONE)
        return

    body = (
        f"[DEMO] New Lead!\n"
        f"Name:   {name}\n"
        f"Phone:  {raw_phone}\n"          # lead's phone — informational only
        f"Course: {lead.get('course') or 'N/A'}\n"
        f"Source: {lead.get('source') or 'N/A'}\n\n"
        f"Welcome message for lead:\n"
        + WELCOME_WHATSAPP_TEXT.format(name=name)
    )

    try:
        def _send():
            client = _get_twilio_client()
            return client.messages.create(
                body  = body,
                from_ = settings.TWILIO_WHATSAPP_FROM,
                to    = f"whatsapp:{admin}",   # ← always ADMIN_PHONE
            ).sid

        sid = await asyncio.to_thread(_send)
        logger.info(
            "whatsapp_sandbox_sent",
            sent_to    = admin,              # where Twilio sent it
            lead_phone = raw_phone,          # lead's submitted phone (informational)
            sid        = sid,
        )

        # Store under ADMIN_PHONE so conversation history lookup works
        sanitised_lead = _sanitise_phone(raw_phone) if raw_phone != "N/A" else admin
        await _store_outbound(sanitised_lead or admin, name, body, "whatsapp", sid)

    except Exception as exc:
        logger.error("whatsapp_sandbox_failed", error=str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
#  EMAIL — SENDGRID (optional)
# ═══════════════════════════════════════════════════════════════════════════════

async def _send_email(lead: dict) -> None:
    if not all([settings.SENDGRID_API_KEY, settings.SENDGRID_FROM_EMAIL]):
        logger.warning("email_skipped_no_credentials")
        return

    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail
    except ImportError:
        logger.warning("sendgrid_not_installed")
        return

    mode     = (settings.WHATSAPP_MODE or "disabled").lower().strip()
    to_email = (lead.get("email") or "").strip() if mode == "production" else ADMIN_EMAIL
    if not to_email:
        return

    name    = lead.get("name") or "there"
    message = Mail(
        from_email   = (settings.SENDGRID_FROM_EMAIL,
                        getattr(settings, "SENDGRID_FROM_NAME", "Invertis Admissions")),
        to_emails    = to_email,
        subject      = WELCOME_EMAIL_SUBJECT,
        html_content = WELCOME_EMAIL_HTML.format(name=name),
    )

    try:
        def _send():
            sg = SendGridAPIClient(settings.SENDGRID_API_KEY)
            return sg.send(message).status_code

        status = await asyncio.to_thread(_send)
        logger.info("email_sent", to=to_email, status=status)
    except Exception as exc:
        logger.error("email_failed", to=to_email, error=str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
#  CHANNEL ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

async def _send_phone_message(lead: dict) -> None:
    mode = (settings.WHATSAPP_MODE or "disabled").lower().strip()
    logger.info("sending_welcome", mode=mode, admin=ADMIN_PHONE)

    if mode == "sms":
        await _send_sms(lead)
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
    await asyncio.gather(
        _send_phone_message(lead),
        # _send_email(lead),   # ← uncomment to enable email
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
        await _store_outbound(lead_phone, lead_name, body, channel, sid)
        return sid

    except Exception as exc:
        logger.error("reply_failed", to=lead_phone, error=str(exc))
        return None