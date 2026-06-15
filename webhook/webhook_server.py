"""
webhook/webhook_server.py — Twilio Inbound WhatsApp/SMS Webhook v7.0
======================================================================
FIXES v7.0
──────────
FIX 1 — All Supabase calls now go through conversation_store._get_supabase()
  which reads from os.environ after load_dotenv — not from config.settings.

FIX 2 — rag_engine import moved to inside function body to prevent
  circular imports and early Supabase init at module load time.

FIX 3 — Added detailed logging at every routing decision point so you
  can see exactly why a message is or isn't going to RAG.

FIX 4 — is_course_query check broadened: "wanna know about" and similar
  conversational phrasings now correctly trigger RAG.

FIX 5 — _process_and_reply now logs the full reply before sending so
  you can diagnose Twilio send failures vs RAG failures separately.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv(override=True)

import asyncio
import urllib.parse
from typing import Optional

import httpx
from fastapi import FastAPI, Form, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from webhook.conversation_store import (
    record_message,
    get_conversation_history,
    normalise_phone,
    phone_variants,
    _get_supabase,
)
from utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

app = FastAPI(
    title       = "Invertis University WhatsApp Webhook",
    description = "RAG-powered admissions assistant v7.0",
    docs_url    = None,
    redoc_url   = None,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TWILIO SIGNATURE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_twilio_signature(request_url: str, post_params: dict, signature: str) -> bool:
    if not settings.TWILIO_AUTH_TOKEN:
        return False
    try:
        from twilio.request_validator import RequestValidator
        validator = RequestValidator(settings.TWILIO_AUTH_TOKEN)
        return validator.validate(request_url, post_params, signature)
    except Exception as exc:
        logger.warning("signature_validation_error", error=str(exc))
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  KEYWORD FALLBACK REPLIES
# ═══════════════════════════════════════════════════════════════════════════════

_GREETING_REPLIES: dict[str, str] = {
    "hi":             "Hello! 👋 How can I help you today? Ask me about courses, fees, or admissions at Invertis University.",
    "hello":          "Hello! 👋 How can I help you today? Ask me about courses, fees, or admissions at Invertis University.",
    "hey":            "Hey there! 👋 Ask me anything about Invertis University — courses, admissions, hostel, or fees!",
    "help":           "I'm here to help! Ask me about courses, fees, admission process, or hostel at Invertis University. 😊",
    "ok":             "Great! Feel free to ask anything about Invertis University. 😊",
    "okay":           "Great! Feel free to ask anything about Invertis University. 😊",
    "thanks":         "You're welcome! 😊 Anything else I can help with?",
    "thank you":      "You're welcome! 😊 Anything else I can help with?",
    "thankyou":       "You're welcome! 😊 Anything else I can help with?",
    "ok thank you":   "You're welcome! 😊 Feel free to ask anything about Invertis University.",
    "okay thank you": "You're welcome! 😊 Feel free to ask anything about Invertis University.",
    "got it":         "Great! Let me know if you have more questions about admissions or courses. 😊",
    "noted":          "Noted! Ask me anything about Invertis University courses or admissions. 😊",
    "ok done":        "Great! Feel free to ask anything about Invertis University. 😊",
    "okay done":      "Great! Feel free to ask anything about Invertis University. 😊",
}

_CHATTER_REPLY = (
    "Glad to hear that! 😊 Feel free to ask me anything about "
    "Invertis University — courses, fees, hostel, or admissions."
)

_DEFAULT_REPLY = (
    "Thank you for your message! 🙏 Our counsellor will get back to you shortly.\n\n"
    "You can also ask me directly about:\n"
    "• Courses & programmes\n"
    "• Fee structure\n"
    "• Admission process\n"
    "• Hostel & campus\n\n"
    "— Invertis University Admissions"
)


def _build_auto_reply(body: str) -> str:
    import re
    cleaned = re.sub(r"[^\w\s]", "", body.lower().strip())

    if cleaned in _GREETING_REPLIES:
        return _GREETING_REPLIES[cleaned]

    for keyword, reply in _GREETING_REPLIES.items():
        if re.search(r"\b" + re.escape(keyword) + r"\b", cleaned):
            return reply

    _positive = re.compile(
        r"\b(nice|great|good|wow|cool|awesome|wonderful|excellent|perfect|super|fantastic|sounds\s+good)\b",
        re.IGNORECASE,
    )
    if _positive.search(body):
        return _CHATTER_REPLY

    return _DEFAULT_REPLY


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTBOUND REPLY via Twilio REST API
# ═══════════════════════════════════════════════════════════════════════════════

async def _send_twilio_reply(
    to_number: str,
    body:      str,
    channel:   str = "whatsapp",
) -> Optional[str]:
    if not all([settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN]):
        logger.warning("reply_skipped_missing_twilio_credentials")
        return None

    if channel == "whatsapp":
        from_number = settings.TWILIO_WHATSAPP_FROM
        to_         = f"whatsapp:{to_number}"
    else:
        from_number = settings.TWILIO_SMS_FROM
        to_         = to_number

    if not from_number:
        logger.warning("reply_skipped_no_from_number", channel=channel)
        return None

    url = (
        f"https://api.twilio.com/2010-04-01/Accounts/"
        f"{settings.TWILIO_ACCOUNT_SID}/Messages.json"
    )

    try:
        async with httpx.AsyncClient(timeout=15) as http:
            resp = await http.post(
                url,
                data = {"From": from_number, "To": to_, "Body": body},
                auth = (settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN),
            )
            if resp.status_code == 201:
                sid = resp.json().get("sid")
                logger.info("reply_sent", to=to_number, channel=channel, sid=sid)
                return sid
            logger.error(
                "reply_failed",
                to=to_number, status=resp.status_code, body=resp.text[:300],
            )
    except Exception as exc:
        logger.error("reply_exception", to=to_number, error=str(exc))

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  LEAD LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════

async def _lookup_lead_by_phone(phone: str) -> Optional[dict]:
    try:
        db       = _get_supabase()
        variants = phone_variants(phone)
        result   = await asyncio.to_thread(
            lambda: db.table("Lead")
            .select("id, name, phone, email, pickedBy, aiScore, type")
            .in_("phone", variants)
            .order("createdAt", desc=True)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        return rows[0] if rows else None
    except Exception as exc:
        logger.warning("lead_lookup_failed", phone=phone, error=str(exc))
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  TWIML RESPONSE
# ═══════════════════════════════════════════════════════════════════════════════

def _twiml_empty() -> Response:
    return Response(
        content     = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
        media_type  = "application/xml",
        status_code = 200,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  DEDUP
# ═══════════════════════════════════════════════════════════════════════════════

_processed_sids: set[str] = set()
_processed_sids_lock      = asyncio.Lock()
_MAX_SID_CACHE            = 10_000


async def _is_duplicate_sid(sid: str) -> bool:
    async with _processed_sids_lock:
        if sid in _processed_sids:
            return True
        if len(_processed_sids) > _MAX_SID_CACHE:
            _processed_sids.clear()
        _processed_sids.add(sid)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  INBOUND WEBHOOK
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/webhook/whatsapp")
async def whatsapp_webhook(
    request:       Request,
    From:          str           = Form(...),
    Body:          str           = Form(""),
    ProfileName:   str           = Form(""),
    MessageSid:    str           = Form(""),
    NumMedia:      str           = Form("0"),
    MediaUrl0:     Optional[str] = Form(None),
    SmsStatus:     Optional[str] = Form(None),
    MessageStatus: Optional[str] = Form(None),
):
    # Ignore delivery status callbacks
    if SmsStatus in ("sent", "delivered", "read", "failed", "undelivered") or \
       MessageStatus in ("sent", "delivered", "read", "failed", "undelivered"):
        logger.debug("status_callback_ignored", status=SmsStatus or MessageStatus)
        return _twiml_empty()

    if not Body.strip() and int(NumMedia or 0) == 0:
        return _twiml_empty()

    if MessageSid and await _is_duplicate_sid(MessageSid):
        logger.warning("duplicate_webhook_ignored", sid=MessageSid)
        return _twiml_empty()

    if getattr(settings, "VALIDATE_TWILIO_SIGNATURE", False):
        sig       = request.headers.get("X-Twilio-Signature", "")
        form_data = dict(await request.form())
        if not _validate_twilio_signature(str(request.url), form_data, sig):
            logger.warning("invalid_twilio_signature", from_=From)
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    raw_phone  = From.replace("whatsapp:", "").strip()
    norm_phone = normalise_phone(raw_phone)
    channel    = "whatsapp" if From.lower().startswith("whatsapp:") else "sms"
    lead_name  = ProfileName.strip() or "Student"

    logger.info(
        "inbound_received",
        from_   = norm_phone,
        channel = channel,
        preview = Body[:80],
        sid     = MessageSid,
    )

    await record_message(
        lead_phone=norm_phone, lead_name=lead_name,
        direction="inbound", body=Body, channel=channel, sid=MessageSid,
    )

    if int(NumMedia or 0) > 0 and MediaUrl0:
        await record_message(
            lead_phone=norm_phone, lead_name=lead_name,
            direction="inbound", body=f"[Media attached: {MediaUrl0}]",
            channel=channel, sid=MessageSid + "_media",
        )

    lead = await _lookup_lead_by_phone(norm_phone)
    if lead:
        display_name = lead.get("name") or lead_name
        logger.info("lead_matched", name=display_name, score=lead.get("aiScore"))
    else:
        display_name = lead_name
        logger.info("lead_not_in_crm_replying_anyway", phone=norm_phone)

    async def _process_and_reply() -> None:
        # Import inside function to avoid circular import at module load
        from webhook.rag_engine import rag_reply, is_course_query, is_pure_greeting

        reply_body: str = ""

        try:
            body_stripped = Body.strip()

            logger.info("routing_decision",
                        phone=norm_phone,
                        is_greeting=is_pure_greeting(body_stripped),
                        is_course=is_course_query(body_stripped),
                        preview=body_stripped[:60])

            if is_pure_greeting(body_stripped):
                logger.info("routing_greeting", preview=body_stripped[:40])
                reply_body = _build_auto_reply(body_stripped)

            elif is_course_query(body_stripped):
                logger.info("routing_to_rag", preview=body_stripped[:60])
                try:
                    reply_body = await asyncio.wait_for(
                        rag_reply(norm_phone, body_stripped, display_name),
                        timeout=30.0,
                    )
                except asyncio.TimeoutError:
                    logger.error("rag_timeout", phone=norm_phone)
                    reply_body = (
                        "I'm looking that up — our counsellor will follow up shortly! 🙏"
                    )
                except Exception as rag_exc:
                    logger.error("rag_exception", phone=norm_phone,
                                 error=str(rag_exc), exc_info=True)
                    reply_body = (
                        "I encountered an issue retrieving that — our counsellor will "
                        "reach out shortly. 🙏"
                    )

            else:
                logger.info("routing_keyword_fallback", preview=body_stripped[:40])
                reply_body = _build_auto_reply(body_stripped)

        except Exception as exc:
            logger.error("process_and_reply_routing_failed", phone=norm_phone,
                         error=str(exc), exc_info=True)
            reply_body = (
                "I encountered an issue — our counsellor will reach out shortly. 🙏"
            )

        if not reply_body:
            logger.error("reply_body_empty_using_fallback", phone=norm_phone)
            reply_body = (
                "Our counsellor will reach out with the information shortly. 🙏"
            )

        logger.info("sending_reply",
                    phone=norm_phone,
                    reply_preview=reply_body[:120])

        try:
            reply_sid = await _send_twilio_reply(norm_phone, reply_body, channel=channel)
            if reply_sid:
                await record_message(
                    lead_phone = norm_phone,
                    lead_name  = display_name,
                    direction  = "outbound",
                    body       = reply_body,
                    channel    = channel,
                    sid        = reply_sid,
                )
            else:
                logger.error("reply_sid_none_message_not_sent", phone=norm_phone)
        except Exception as send_exc:
            logger.error("send_reply_exception", phone=norm_phone, error=str(send_exc))

    task = asyncio.create_task(
        _process_and_reply(),
        name=f"reply-{norm_phone}-{MessageSid[:8] if MessageSid else 'nosid'}",
    )

    def _task_done_callback(t: asyncio.Task) -> None:
        if t.cancelled():
            logger.warning("reply_task_cancelled", phone=norm_phone)
            return
        exc = t.exception()
        if exc:
            logger.error("reply_task_unhandled_exception",
                         phone=norm_phone, error=str(exc))

    task.add_done_callback(_task_done_callback)
    return _twiml_empty()


# ═══════════════════════════════════════════════════════════════════════════════
#  RAG TEST ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/rag-test")
async def rag_test(
    q:     str = Query(..., description="Question to test"),
    name:  str = Query("TestStudent", description="Student name"),
    phone: str = Query("test_phone", description="Phone for history context"),
):
    from webhook.rag_engine import (
        rag_reply, is_course_query, is_pure_greeting,
        _embedder, _get_db, _TOP_K, _get_similarity_threshold,
    )

    is_greeting = is_pure_greeting(q)
    is_course   = is_course_query(q)

    logger.info("rag_test_request",
                q=q[:60], is_greeting=is_greeting, is_course=is_course)

    if is_greeting:
        return JSONResponse({
            "query":   q,
            "routing": "greeting",
            "reply":   _build_auto_reply(q),
        })

    if not is_course:
        return JSONResponse({
            "query":           q,
            "is_course_query": False,
            "routing":         "keyword_fallback",
            "reply":           _build_auto_reply(q),
            "hint":            "Message not detected as a course query. Check is_course_query() logic.",
        })

    q_vec = await asyncio.to_thread(lambda: _embedder.encode(q).tolist())

    raw_chunks: list[dict] = []
    try:
        db     = _get_db()
        result = db.rpc("match_course_chunks", {
            "query_embedding": q_vec,
            "match_count":     _TOP_K,
        }).execute()
        raw_chunks = result.data or []
    except Exception as exc:
        logger.error("rag_test_rpc_error", error=str(exc))
        raw_chunks = []

    reply = await rag_reply(phone, q, name)
    min_sim, _ = _get_similarity_threshold(q)

    return JSONResponse({
        "query":                q,
        "is_course_query":      True,
        "routing":              "rag",
        "similarity_threshold": min_sim,
        "chunks_retrieved": [
            {
                "course_name":     r.get("course_name"),
                "similarity":      round(float(r.get("similarity", 0)), 4),
                "above_threshold": float(r.get("similarity", 0)) >= min_sim,
                "chunk_preview":   r.get("chunk_text", "")[:200],
            }
            for r in raw_chunks
        ],
        "reply": reply,
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  MANUAL REPLY ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

class ReplyRequest(BaseModel):
    to_phone:  str
    body:      str
    lead_name: str           = "Lead"
    channel:   str           = "whatsapp"
    lead_id:   Optional[str] = None


@app.post("/send-reply")
async def send_reply_endpoint(req: ReplyRequest):
    if not req.to_phone or not req.body:
        raise HTTPException(status_code=400, detail="to_phone and body are required")

    sid = await _send_twilio_reply(req.to_phone, req.body, channel=req.channel)
    if not sid:
        raise HTTPException(
            status_code=500,
            detail="Failed to send — check Twilio credentials and server logs",
        )

    await record_message(
        lead_phone = req.to_phone,
        lead_name  = req.lead_name,
        direction  = "outbound",
        body       = req.body,
        channel    = req.channel,
        sid        = sid,
    )
    return JSONResponse({"status": "sent", "twilio_sid": sid, "to": req.to_phone})


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVERSATION HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/conversation/{phone}")
async def get_history(phone: str, limit: int = 50):
    decoded  = urllib.parse.unquote(phone)
    messages = await get_conversation_history(decoded, limit=limit)
    return {"phone": decoded, "count": len(messages), "messages": messages}


# ═══════════════════════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    import os
    status = {
        "service":    "Invertis WhatsApp Webhook v7.0",
        "status":     "ok",
        "components": {},
    }

    try:
        db = _get_supabase()
        db.table("Lead").select("id").limit(1).execute()
        status["components"]["supabase"] = "ok"
    except Exception as exc:
        status["components"]["supabase"] = f"error: {str(exc)[:80]}"
        status["status"] = "degraded"

    try:
        db     = _get_supabase()
        result = db.table("CourseChunk").select("id", count="exact").execute()
        count  = result.count or 0
        status["components"]["course_chunks"] = f"{count} chunks indexed"
        if count == 0:
            status["components"]["course_chunks"] = "WARNING: 0 chunks — run ingest_courses.py"
            status["status"] = "degraded"
    except Exception as exc:
        status["components"]["course_chunks"] = f"error: {str(exc)[:80]}"

    has_twilio = bool(settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN)
    status["components"]["twilio"] = "configured" if has_twilio else "missing credentials"

    has_groq = bool(os.environ.get("GROQ_API_KEY", ""))
    status["components"]["groq"] = "configured" if has_groq else "missing GROQ_API_KEY"

    return JSONResponse(status)