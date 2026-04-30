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

FIXES (v2.1)
────────────
  • Removed logging.basicConfig() call.
    StructuredLogger (utils/logger.py) self-configures each module's
    handler on first use. Calling basicConfig() after module import
    causes duplicate log lines because root-logger and module-loggers
    both fire. Let StructuredLogger own all log configuration.
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

# Load .env FIRST — before any module that reads settings at import time
load_dotenv()

from agent.lead_ranking_agent import main as agent_main
from agent.welcome_service import validate_twilio_config
from utils.logger import get_logger

logger = get_logger("main")


async def main() -> None:
    print("=" * 65)
    print("  Invertis — Lead Ranking Agent  |  Production v2.1")
    print("=" * 65)
    print()
    print("  Webhook server (two-way WhatsApp) runs separately:")
    print("  → uvicorn webhook.webhook_server:app --port 8000")
    print()
    print("=" * 65)

    # Validate Twilio credentials at startup — surfaces errors before first lead
    validate_twilio_config()

    # Start the ranking agent listen loop
    await agent_main()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        import traceback
        print("\n🔥 REAL ERROR BELOW:\n")
        traceback.print_exc()
        sys.exit(1)