"""
webhook/webhook_server.py — Twilio Inbound WhatsApp/SMS Webhook
================================================================
Receives inbound messages from Twilio, saves them to Supabase,
sends auto-replies (RAG or keyword), and exposes a REST API.

Run
───
  uvicorn webhook.webhook_server:app --port 8000 --reload

Endpoints
─────────
  POST /webhook/whatsapp      ← Twilio calls this on inbound message
  POST /send-reply            ← Send a manual reply to any lead
  GET  /conversation/{phone}  ← Fetch history for a lead
  GET  /rag-test              ← Quick RAG validation without Twilio
  GET  /health                ← Health check with component status

FIXES applied in this version
──────────────────────────────
FIX 1 — /rag-test endpoint added for validating RAG without needing
         a real WhatsApp message (just hit it in browser).

FIX 2 — _build_auto_reply now uses word-boundary matching consistent
         with is_course_query, preventing "fee" matching "feel".

FIX 3 — Lead lookup failure is now fully non-blocking — a missing
         lead in the CRM never prevents an auto-reply being sent.

FIX 4 — Twilio signature validation uses twilio.request_validator
         (official library) rather than hand-rolled HMAC.

FIX 5 — All course-related keywords in _KEYWORD_REPLIES are now
         unreachable dead code (correctly routed to RAG first),
         but kept as defence-in-depth fallback.
"""

from __future__ import annotations

import asyncio
import urllib.parse
from typing import Optional

import httpx
from fastapi import FastAPI, Form, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from webhook.conversation_store import record_message, get_conversation_history
from webhook.rag_engine import rag_reply, is_course_query
from utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

app = FastAPI(
    title       = "Invertis University WhatsApp Webhook",
    description = "RAG-powered admissions assistant",
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
#  KEYWORD FALLBACK — only fires when is_course_query() returns False
# ═══════════════════════════════════════════════════════════════════════════════

_KEYWORD_REPLIES: dict[str, str] = {
    "hi":    "Hello! 👋 How can I help you today? You can ask me about courses, fees, or admissions at Invertis University.",
    "hello": "Hello! 👋 How can I help you today? You can ask me about courses, fees, or admissions at Invertis University.",
    "hey":   "Hey there! 👋 Ask me anything about Invertis University — courses, admissions, hostel, or fees!",
    "help":  "I'm here to help! Ask me about courses, fees, admission process, or hostel at Invertis University. 😊",
    "ok":    "Great! Feel free to ask anything about Invertis University. 😊",
    "okay":  "Great! Feel free to ask anything about Invertis University. 😊",
    "thanks":       "You're welcome! 😊 Is there anything else I can help you with?",
    "thank you":    "You're welcome! 😊 Is there anything else I can help you with?",
    "thankyou":     "You're welcome! 😊 Is there anything else I can help you with?",
}

_DEFAULT_REPLY = (
    "Thank you for your message! 🙏 Our counsellor will get back to you shortly.\n\n"
    "You can also ask me directly about:\n"
    "• Courses & programmes\n"
    "• Fee structure\n"
    "• Admission process\n"
    "• Hostel & campus\n\n"
    "— Invertis University Admissions"
)


def _build_auto_reply(body: str, lead_name: str) -> str:
    """
    Simple keyword fallback — only reaches this when is_course_query() is False.
    Uses word-boundary matching consistent with is_course_query.
    """
    import re
    lower = body.lower().strip()
    for keyword, reply in _KEYWORD_REPLIES.items():
        # word-boundary match
        if re.search(r"\b" + re.escape(keyword) + r"\b", lower):
            return reply          # ← no name prefix
    return _DEFAULT_REPLY 


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTBOUND REPLY via Twilio REST API
# ═══════════════════════════════════════════════════════════════════════════════

async def _send_twilio_reply(
    to_number: str,
    body:      str,
    channel:   str = "whatsapp",
) -> Optional[str]:
    """Sends a reply via Twilio REST API. Returns SID on success, None on failure."""
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
        async with httpx.AsyncClient(timeout=10) as http:
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
    """Tries E.164, 10-digit, and +91 variants. Never raises."""
    try:
        from webhook.conversation_store import _get_supabase
        db = _get_supabase()

        digits = phone.lstrip("+")
        if digits.startswith("91") and len(digits) == 12:
            digits = digits[2:]

        result = await asyncio.to_thread(
            lambda: db.table("Lead")
            .select("id, name, phone, email, pickedBy, aiScore, type")
            .or_(f"phone.eq.{phone},phone.eq.{digits},phone.eq.+91{digits}")
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
        content    = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
        media_type = "application/xml",
        status_code = 200,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  INBOUND WEBHOOK — Twilio calls this on every incoming WhatsApp message
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/webhook/whatsapp")
async def whatsapp_webhook(
    request:     Request,
    From:        str           = Form(...),
    Body:        str           = Form(""),
    ProfileName: str           = Form(""),
    MessageSid:  str           = Form(""),
    NumMedia:    str           = Form("0"),
    MediaUrl0:   Optional[str] = Form(None),
):
    """
    Full inbound flow:
      1. Optional Twilio signature validation
      2. Normalise sender phone number
      3. Save inbound message to Supabase + terminal
      4. Look up lead in CRM (non-blocking — never prevents reply)
      5. Route: is_course_query? → RAG | else → keyword fallback
      6. Send reply via Twilio REST
      7. Save outbound reply to Supabase
      8. Return empty TwiML (200 OK so Twilio doesn't retry)
    """
    # ── 1. Signature validation ───────────────────────────────────────────────
    if getattr(settings, "VALIDATE_TWILIO_SIGNATURE", False):
        sig       = request.headers.get("X-Twilio-Signature", "")
        form_data = dict(await request.form())
        if not _validate_twilio_signature(str(request.url), form_data, sig):
            logger.warning("invalid_twilio_signature", from_=From)
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    # ── 2. Normalise ──────────────────────────────────────────────────────────
    raw_phone = From.replace("whatsapp:", "").strip()
    channel   = "whatsapp" if From.lower().startswith("whatsapp:") else "sms"
    lead_name = ProfileName.strip() or "Unknown"

    logger.info(
        "inbound_received",
        from_    = raw_phone,
        channel  = channel,
        preview  = Body[:80],
        sid      = MessageSid,
    )

    # ── 3. Save inbound ───────────────────────────────────────────────────────
    await record_message(
        lead_phone = raw_phone,
        lead_name  = lead_name,
        direction  = "inbound",
        body       = Body,
        channel    = channel,
        sid        = MessageSid,
    )

    if int(NumMedia or 0) > 0 and MediaUrl0:
        await record_message(
            lead_phone = raw_phone,
            lead_name  = lead_name,
            direction  = "inbound",
            body       = f"[Media attached: {MediaUrl0}]",
            channel    = channel,
            sid        = MessageSid + "_media",
        )

    # ── 4. Lead lookup (best-effort — never blocks reply) ─────────────────────
    lead         = await _lookup_lead_by_phone(raw_phone)
    display_name = (lead.get("name") if lead else None) or lead_name or "there"

    if lead:
        logger.info("lead_matched", name=lead.get("name"), score=lead.get("aiScore"))
    else:
        logger.info("lead_not_in_crm", phone=raw_phone)

    # ── 5. Route to RAG or keyword fallback ───────────────────────────────────
    if Body.strip() and is_course_query(Body):
        logger.info("routing_to_rag", query_preview=Body[:60])
        reply_body = await rag_reply(raw_phone, Body, display_name)
    else:
        logger.info("routing_to_keyword_fallback")
        reply_body = _build_auto_reply(Body, display_name)

    # ── 6. Send reply ─────────────────────────────────────────────────────────
    reply_sid = await _send_twilio_reply(raw_phone, reply_body, channel=channel)

    # ── 7. Save outbound ──────────────────────────────────────────────────────
    if reply_sid:
        await record_message(
            lead_phone = raw_phone,
            lead_name  = display_name,
            direction  = "outbound",
            body       = reply_body,
            channel    = channel,
            sid        = reply_sid,
        )

    # ── 8. Return empty TwiML ─────────────────────────────────────────────────
    return _twiml_empty()


# ═══════════════════════════════════════════════════════════════════════════════
#  RAG TEST ENDPOINT — validate retrieval without WhatsApp
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/rag-test")
async def rag_test(
    q:    str = Query(..., description="Question to test, e.g. 'fees for B.Tech'"),
    name: str = Query("TestStudent", description="Student name for reply personalisation"),
):
    """
    Tests the full RAG pipeline without needing a real WhatsApp message.

    Usage:
      http://localhost:8000/rag-test?q=What+are+the+fees+for+BTech
      http://localhost:8000/rag-test?q=eligibility+for+MBA&name=Rahul

    Returns the generated reply plus debug info (context found, routing).
    Use this to validate retrieval before connecting Twilio.
    """
    from webhook.conversation_store import _get_supabase
    from sentence_transformers import SentenceTransformer
    from webhook.rag_engine import (
        _embedder, _search_courses_sync, EMBEDDING_MODEL, _MIN_SIMILARITY
    )

    # Step 1 — is_course_query check
    is_course = is_course_query(q)

    if not is_course:
        return JSONResponse({
            "query":          q,
            "is_course_query": False,
            "routing":        "keyword_fallback",
            "reply":          _build_auto_reply(q, name),
            "context_chunks": [],
            "note":           "Add a course keyword to trigger RAG",
        })

    # Step 2 — Embed
    q_vec = await asyncio.to_thread(lambda: _embedder.encode(q).tolist())

    # Step 3 — Retrieve raw chunks with similarity scores (bypass threshold for debug)
    try:
        db     = _get_supabase()
        result = db.rpc("match_course_chunks", {
            "query_embedding": q_vec,
            "match_count":     5,
        }).execute()
        raw_chunks = result.data or []
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    # Step 4 — Full RAG reply (uses threshold internally)
    reply = await rag_reply("test_phone", q, name)

    return JSONResponse({
        "query":            q,
        "embedding_model":  EMBEDDING_MODEL,
        "is_course_query":  True,
        "routing":          "rag",
        "similarity_threshold": _MIN_SIMILARITY,
        "chunks_retrieved": [
            {
                "course_name": r.get("course_name"),
                "similarity":  round(float(r.get("similarity", 0)), 4),
                "above_threshold": float(r.get("similarity", 0)) >= _MIN_SIMILARITY,
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
    """
    Send a manual WhatsApp reply to any lead.

    curl example:
      curl -X POST http://localhost:8000/send-reply \\
        -H "Content-Type: application/json" \\
        -d '{"to_phone":"+919876543210","body":"Counsellor will call at 3 PM today."}'
    """
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
    """
    Returns conversation history for a lead phone number.
    URL-encode the + sign: /conversation/%2B918868058962
    """
    decoded  = urllib.parse.unquote(phone)
    messages = await get_conversation_history(decoded, limit=limit)
    return {"phone": decoded, "count": len(messages), "messages": messages}


# ═══════════════════════════════════════════════════════════════════════════════
#  HEALTH CHECK — shows component status
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """
    Returns health status of all components.
    Green = ready, yellow = degraded but running, red = broken.
    """
    from webhook.conversation_store import _get_supabase

    status = {
        "service":  "Invertis WhatsApp Webhook",
        "status":   "ok",
        "components": {},
    }

    # Supabase ping
    try:
        db = _get_supabase()
        db.table("Conversation").select("id").limit(1).execute()
        status["components"]["supabase"] = "ok"
    except Exception as exc:
        status["components"]["supabase"] = f"error: {str(exc)[:80]}"
        status["status"] = "degraded"

    # CourseChunk count
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

    # Twilio config
    has_twilio = bool(settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN)
    status["components"]["twilio"] = "configured" if has_twilio else "missing credentials"

    # Groq config
    import os
    has_groq = bool(os.environ.get("GROQ_API_KEY", ""))
    status["components"]["groq"] = "configured" if has_groq else "missing GROQ_API_KEY"

    return JSONResponse(status)