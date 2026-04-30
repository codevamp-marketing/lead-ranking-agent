# ─── Stage 1: build dependencies ────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install CPU-only torch FIRST (avoid CUDA bloat)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu

# Then install rest
RUN pip install --no-cache-dir -r requirements.txt


# ─── Stage 2: runtime image ──────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code
COPY agent/     ./agent/
COPY scoring/   ./scoring/
COPY config/    ./config/
COPY utils/     ./utils/
COPY webhook/   ./webhook/
COPY main.py    .
COPY ingest_courses.py .
COPY validate_rag.py .

# Create non-root user
RUN useradd --no-create-home --shell /bin/false agent
USER agent

CMD ["python", "main.py"]