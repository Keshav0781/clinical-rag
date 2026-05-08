# ─────────────────────────────────────────────
# Stage 1 — Builder
# Installs all dependencies including build tools
# ─────────────────────────────────────────────
FROM python:3.11.9-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─────────────────────────────────────────────
# Stage 2 — Runtime
# Clean image, no build tools, just what runs
# ─────────────────────────────────────────────


FROM python:3.11.9-slim

WORKDIR /app

# Security — never run as root in production
RUN useradd -m -u 1000 appuser
RUN apt-get update && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application files
COPY src/ ./src/
COPY data/ ./data/
COPY documents/ ./documents/

# Give appuser ownership of app files
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# Health check — Cloud Run and monitoring systems use this
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
