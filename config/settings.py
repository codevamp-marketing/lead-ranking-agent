"""
config/settings.py — Centralized configuration.
All settings read from .env — no hardcoded values anywhere else.

FIXES (v2.1)
────────────
  • CRM_API_BASE is now optional (empty string allowed). The agent has a
    3-tier write fallback (CRM → Supabase → DLQ), so CRM is not required
    to start. Missing CRM is logged as a warning, not a startup crash.
  • DATABASE_URL is still required (Postgres LISTEN/NOTIFY needs it).
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _clean(val: str | None) -> str:
    if not val:
        return ""
    return val.strip().strip('"').strip("'")


class _Settings:
    def __init__(self):
        # ── Core (required) ───────────────────────────────────────────────────
        self.SUPABASE_URL         = _clean(os.getenv("SUPABASE_URL"))
        self.SUPABASE_SERVICE_KEY = _clean(os.getenv("SUPABASE_SERVICE_KEY"))
        self.DATABASE_URL         = _clean(os.getenv("DATABASE_URL"))

        # FIX: CRM_API_BASE is optional — agent can run Supabase-only
        self.CRM_API_BASE         = _clean(os.getenv("CRM_API_BASE", "")).rstrip("/")

        # ── Twilio ────────────────────────────────────────────────────────────
        self.TWILIO_ACCOUNT_SID   = _clean(os.getenv("TWILIO_ACCOUNT_SID"))
        self.TWILIO_AUTH_TOKEN    = _clean(os.getenv("TWILIO_AUTH_TOKEN"))
        self.TWILIO_WHATSAPP_FROM = _clean(os.getenv("TWILIO_WHATSAPP_FROM"))
        self.TWILIO_SMS_FROM      = _clean(os.getenv("TWILIO_SMS_FROM"))
        self.WHATSAPP_MODE        = _clean(os.getenv("WHATSAPP_MODE", "disabled"))

        # ── SendGrid ──────────────────────────────────────────────────────────
        self.SENDGRID_API_KEY     = _clean(os.getenv("SENDGRID_API_KEY"))
        self.SENDGRID_FROM_EMAIL  = _clean(os.getenv("SENDGRID_FROM_EMAIL"))
        self.SENDGRID_FROM_NAME   = _clean(os.getenv("SENDGRID_FROM_NAME", "Invertis Admissions"))

        # ── Webhook server ────────────────────────────────────────────────────
        self.WEBHOOK_HOST               = _clean(os.getenv("WEBHOOK_HOST", "0.0.0.0"))
        self.WEBHOOK_PORT               = int(os.getenv("WEBHOOK_PORT", "8000"))
        self.VALIDATE_TWILIO_SIGNATURE  = os.getenv("VALIDATE_TWILIO_SIGNATURE", "false").lower() == "true"

        # ── Agent tuning ──────────────────────────────────────────────────────
        self.PG_NOTIFY_CHANNEL  = "new_lead"
        self.PG_SELECT_TIMEOUT  = 30    # FIX: was 5s — 30s reduces busy-loop under low traffic
        self.CRM_TIMEOUT        = 15
        self.CRM_RETRY_ATTEMPTS = 3
        self.LOG_LEVEL          = _clean(os.getenv("LOG_LEVEL", "INFO"))

        # ── Fail fast on truly required vars ─────────────────────────────────
        # CRM_API_BASE intentionally excluded — it is optional
        required = ("SUPABASE_URL", "SUPABASE_SERVICE_KEY", "DATABASE_URL")
        missing  = [name for name in required if not getattr(self, name)]
        if missing:
            raise EnvironmentError(
                f"\n\nMissing required environment variables: {', '.join(missing)}\n"
                "Check your .env file.\n"
            )

        # Warn (don't crash) if CRM is not configured
        if not self.CRM_API_BASE:
            import warnings
            warnings.warn(
                "CRM_API_BASE not set — agent will write directly to Supabase (tier 2).",
                RuntimeWarning,
                stacklevel=2,
            )


settings = _Settings()