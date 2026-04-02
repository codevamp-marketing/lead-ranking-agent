"""
utils/logger.py — Clean, readable logging
==========================================
Two modes controlled by LOG_FORMAT env var:

  LOG_FORMAT=pretty (default for local dev)
    12:52:22  INFO   ✅ lead_ranked        gaurav | Manual → Warm (53) | ₹1,59,000
    12:52:22  WARN   ⚠ scoring_rules      StreamReset — falling back to signals

  LOG_FORMAT=json (for production / log aggregators)
    {"time": "...", "level": "INFO", "event": "lead_ranked", ...}

Only INFO+ is shown by default. DEBUG lines are suppressed unless
LOG_LEVEL=DEBUG is set.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone


# ── Pretty formatter (local dev) ──────────────────────────────────────────────

_ICONS = {
    "DEBUG":    "🔍",
    "INFO":     "  ",
    "WARNING":  "⚠ ",
    "ERROR":    "✗ ",
    "CRITICAL": "💀",
}

class _PrettyFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        now   = datetime.now().strftime("%H:%M:%S")
        level = record.levelname[:4]
        event = record.getMessage()
        icon  = _ICONS.get(record.levelname, "  ")

        extra = getattr(record, "_structured", {})
        # Format extra fields as  key=value  pairs on the same line
        kv = "  ".join(f"{k}={v}" for k, v in extra.items()) if extra else ""

        line = f"{now}  {level:<4}  {icon} {event:<22}"
        if kv:
            line += f"  {kv}"
        return line


# ── JSON formatter (production) ───────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "time":  datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        out.update(getattr(record, "_structured", {}))
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


# ── StructuredLogger ──────────────────────────────────────────────────────────

class StructuredLogger:
    """
    Wraps stdlib Logger. Accepts keyword arguments as structured fields.

    Usage:
        logger.info("lead_ranked", name="Rahul", score=85, type="Hot")
        logger.warning("crm_retry", attempt=2, wait_sec=3.1)
        logger.error("crm_failed", lead_id="abc", error="timeout")
    """

    def __init__(self, name: str):
        self._log = logging.getLogger(name)
        if not self._log.handlers:
            fmt = os.getenv("LOG_FORMAT", "pretty").lower()
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(_JsonFormatter() if fmt == "json" else _PrettyFormatter())
            self._log.addHandler(handler)

            level_name = os.getenv("LOG_LEVEL", "INFO").upper()
            self._log.setLevel(getattr(logging, level_name, logging.INFO))
            self._log.propagate = False

    def _emit(self, level: int, event: str, **kwargs):
        if self._log.isEnabledFor(level):
            record = self._log.makeRecord(
                self._log.name, level, "(unknown)", 0, event, (), None,
            )
            record._structured = kwargs
            self._log.handle(record)

    def debug(self, event: str, **kwargs):    self._emit(logging.DEBUG,    event, **kwargs)
    def info(self, event: str, **kwargs):     self._emit(logging.INFO,     event, **kwargs)
    def warning(self, event: str, **kwargs):  self._emit(logging.WARNING,  event, **kwargs)
    def error(self, event: str, **kwargs):    self._emit(logging.ERROR,    event, **kwargs)
    def critical(self, event: str, **kwargs): self._emit(logging.CRITICAL, event, **kwargs)


def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger(name)