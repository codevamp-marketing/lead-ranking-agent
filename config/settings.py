import os
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE     = _PROJECT_ROOT / ".env"

load_dotenv(_ENV_FILE, override=True)

def _clean(val: str | None) -> str:
    if not val:
        return ""
    return val.strip().strip('"').strip("'")

class _Settings:
    def __init__(self):
        self.SUPABASE_URL         = _clean(os.getenv("SUPABASE_URL"))
        self.SUPABASE_SERVICE_KEY = _clean(os.getenv("SUPABASE_SERVICE_KEY"))
        self.DATABASE_URL         = _clean(os.getenv("DATABASE_URL"))
        self.CRM_API_BASE         = _clean(os.getenv("CRM_API_BASE", "")).rstrip("/")
        self.TWILIO_ACCOUNT_SID   = _clean(os.getenv("TWILIO_ACCOUNT_SID"))
        self.TWILIO_AUTH_TOKEN    = _clean(os.getenv("TWILIO_AUTH_TOKEN"))
        self.TWILIO_WHATSAPP_FROM = _clean(os.getenv("TWILIO_WHATSAPP_FROM"))
        self.TWILIO_SMS_FROM      = _clean(os.getenv("TWILIO_SMS_FROM"))
        self.WHATSAPP_MODE        = _clean(os.getenv("WHATSAPP_MODE", "disabled"))
        self.ADMIN_PHONE          = _clean(os.getenv("ADMIN_PHONE", "+919634776903"))
        self.GROQ_API_KEY         = _clean(os.getenv("GROQ_API_KEY"))
        self.SENDGRID_API_KEY     = _clean(os.getenv("SENDGRID_API_KEY"))
        self.SENDGRID_FROM_EMAIL  = _clean(os.getenv("SENDGRID_FROM_EMAIL"))
        self.SENDGRID_FROM_NAME   = _clean(os.getenv("SENDGRID_FROM_NAME", "Invertis Admissions"))
        self.WEBHOOK_HOST               = _clean(os.getenv("WEBHOOK_HOST", "0.0.0.0"))
        self.WEBHOOK_PORT               = int(os.getenv("WEBHOOK_PORT", "8000"))
        self.VALIDATE_TWILIO_SIGNATURE  = os.getenv("VALIDATE_TWILIO_SIGNATURE", "false").lower() == "true"
        self.PG_NOTIFY_CHANNEL  = "new_lead"
        self.PG_SELECT_TIMEOUT  = 30
        self.CRM_TIMEOUT        = 15
        self.CRM_RETRY_ATTEMPTS = 3
        self.LOG_LEVEL          = _clean(os.getenv("LOG_LEVEL", "INFO"))

        required = ("SUPABASE_URL", "SUPABASE_SERVICE_KEY", "DATABASE_URL")
        missing  = [name for name in required if not getattr(self, name)]
        if missing:
            raise EnvironmentError(f"Missing env vars: {', '.join(missing)}\n.env path: {_ENV_FILE}\nexists: {_ENV_FILE.exists()}")

        if not self.CRM_API_BASE:
            import warnings
            warnings.warn("CRM_API_BASE not set — writing directly to Supabase.", RuntimeWarning, stacklevel=2)

settings = _Settings()
