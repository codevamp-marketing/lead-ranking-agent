"""
config/settings.py — Centralized configuration using python-dotenv only.
No pydantic-settings required — works with your existing requirements.txt.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _clean(val: str | None) -> str:
    """Strip surrounding whitespace and quotes that people often add in .env files."""
    if not val:
        return ""
    return val.strip().strip('"').strip("'")


class _Settings:
    def __init__(self):
        self.SUPABASE_URL         = _clean(os.getenv("SUPABASE_URL"))
        self.SUPABASE_SERVICE_KEY = _clean(os.getenv("SUPABASE_SERVICE_KEY"))
        self.CRM_API_BASE         = _clean(os.getenv("CRM_API_BASE", "")).rstrip("/")
        self.DATABASE_URL         = _clean(os.getenv("DATABASE_URL"))

        # Tunable defaults — override in .env if needed
        self.PG_NOTIFY_CHANNEL  = "new_lead"
        self.PG_SELECT_TIMEOUT  = 5    # seconds
        self.CRM_TIMEOUT        = 15   # seconds
        self.CRM_RETRY_ATTEMPTS = 3
        self.LOG_LEVEL          = "INFO"

        # Validate required vars at startup — fail fast with a clear message
        missing = [
            name for name in
            ("SUPABASE_URL", "SUPABASE_SERVICE_KEY", "CRM_API_BASE", "DATABASE_URL")
            if not getattr(self, name)
        ]
        if missing:
            raise EnvironmentError(
                f"\n\nMissing required environment variables: {', '.join(missing)}\n"
                "Check your .env file and make sure these keys are set.\n"
            )


settings = _Settings()