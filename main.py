"""
main.py — Entry Point for Invertis Lead Ranking Agent
=======================================================
This is the ONLY file you run. It starts the lead ranking agent.

The webhook server runs as a SEPARATE process:
  uvicorn webhook.webhook_server:app --port 8000

Why separate?
  The ranking agent is a long-running asyncio LISTEN loop.
  The webhook server is an HTTP server (FastAPI/uvicorn).
  Mixing both in one process complicates signal handling and logging.
  Two terminals = two clean, independently restartable processes.

Run
---
  # Terminal 1 — ranking agent (this file)
  python main.py

  # Terminal 2 — webhook server (two-way WhatsApp)
  uvicorn webhook.webhook_server:app --port 8000 --reload

  # Terminal 3 — ngrok (exposes webhook to Twilio)
  ngrok http 8000


"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

# ── Load .env FIRST — before ANY module that reads settings at import time ────
# This MUST be the first statement after stdlib imports.
# If you move it below the project imports, settings.py reads empty strings.
load_dotenv(override=True)

# ── Project imports come AFTER load_dotenv() ──────────────────────────────────
from agent.lead_ranking_agent import main as agent_main
from agent.welcome_service import validate_twilio_config
from utils.logger import get_logger

logger = get_logger("main")


# ═══════════════════════════════════════════════════════════════════════════════
#  STARTUP ENV GUARD
#  Must live INSIDE an async def (or a regular def called after load_dotenv).
#  At module level, load_dotenv() has not run yet, so os.environ.get()
#  returns None for every key — causing a false "missing" failure or NameError.
# ═══════════════════════════════════════════════════════════════════════════════

def _check_required_env() -> None:
    """
    Validates that all required environment variables are present.
    Exits immediately with a clear error if any are missing.
    Call this AFTER load_dotenv() has been called (i.e. inside main()).

    Why not at module level?
      load_dotenv() populates os.environ at runtime.
      Module-level code runs when Python imports the file — before
      load_dotenv() is called — so os.environ.get() returns None
      for every .env key, making the guard always fail.
    """
    required = [
        "SUPABASE_URL",
        "SUPABASE_SERVICE_KEY",
        "DATABASE_URL",
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "ADMIN_PHONE",
        "WHATSAPP_MODE",
        "GROQ_API_KEY",
    ]

    missing = [key for key in required if not os.environ.get(key, "").strip()]

    if missing:
        # Use print here — logger may not be fully initialised yet
        print("\n" + "=" * 65)
        print("  ❌  STARTUP BLOCKED — missing environment variables")
        print("=" * 65)
        for key in missing:
            print(f"      → {key}  is not set in .env")
        print()
        print("  Fix: open .env and add the missing values, then restart.")
        print("=" * 65 + "\n")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    # ── 1. Guard: all required env vars must be present ───────────────────────
    # Called here — after load_dotenv() at module level — so .env is loaded.
    _check_required_env()

    # ── 2. Banner ──────────────────────────────────────────────────────────────
    print("=" * 65)
    print("  Invertis — Lead Ranking Agent  |  Production v2.2")
    print("=" * 65)
    print()
    print("  Webhook server (two-way WhatsApp) runs separately:")
    print("  → uvicorn webhook.webhook_server:app --port 8000")
    print()
    print("=" * 65)

    # ── 3. Log startup config so you can verify .env was loaded correctly ──────
    whatsapp_mode = os.environ.get("WHATSAPP_MODE", "NOT SET")
    admin_phone   = os.environ.get("ADMIN_PHONE",   "NOT SET")
    crm_base      = os.environ.get("CRM_API_BASE",  "disabled")

    logger.info(
        "startup_config",
        whatsapp_mode = whatsapp_mode,
        admin_phone   = admin_phone,
        crm_base      = crm_base,
    )

    # Warn immediately if mode is unexpected — saves debugging later
    if whatsapp_mode not in ("sandbox", "production", "sms", "disabled"):
        logger.warning(
            "unknown_whatsapp_mode",
            mode=whatsapp_mode,
            hint="Expected one of: sandbox | production | sms | disabled",
        )

    # ── 4. Validate Twilio credentials before first lead arrives ───────────────
    # Surfaces misconfiguration before the first real lead — not mid-conversation.
    validate_twilio_config()

    # ── 5. Start the ranking agent listen loop ─────────────────────────────────
    await agent_main()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("agent_stopped_by_user")
    except Exception as exc:
        import traceback
        print("\n🔥  UNHANDLED EXCEPTION — full traceback below:\n")
        traceback.print_exc()
        sys.exit(1)