# ─── Stage 1: build dependencies ────────────────────────────────────────────
# Use slim for smaller image. Python 3.11 is stable & well-tested with psycopg2.
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system deps needed to compile psycopg2-binary
# (libpq-dev only needed if using psycopg2 source; binary skips this,
#  but keeping it as a safety net for any native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY lead-ranking-agent/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ─── Stage 2: runtime image ──────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder (keeps runtime image clean)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application packages
# Each directory is a Python package — all must be present for imports to work
COPY lead-ranking-agent/agent/     ./agent/
COPY lead-ranking-agent/scoring/   ./scoring/
COPY lead-ranking-agent/config/    ./config/
COPY lead-ranking-agent/utils/     ./utils/
COPY lead-ranking-agent/.env       ./.env

# ── WORKDIR is /app, so Python resolves:  ────────────────────────────────────
#   from scoring.engine import ...   → /app/scoring/engine.py  ✓
#   from config.settings import ...  → /app/config/settings.py ✓
#   from utils.logger import ...     → /app/utils/logger.py    ✓

# Run as non-root for security (never run agents as root in production)
RUN useradd --no-create-home --shell /bin/false agent
USER agent

# Entry point: run as module so Python sets sys.path to /app correctly
# This is equivalent to: cd /app && python -m agent.lead_ranking_agent
CMD ["python", "-m", "agent.lead_ranking_agent"]