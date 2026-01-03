# ============================================================================
# EasyData Tier-2 — Final Dockerfile (Backend + UI)
# Python 3.11 | Debian Bookworm
# ============================================================================

FROM python:3.11-slim-bookworm

# ----------------------------------------------------------------------------
# 1. Runtime Environment
# ----------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# ----------------------------------------------------------------------------
# 2. System Dependencies (Minimal & Required)
# ----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# 3. Non-Root User
# ----------------------------------------------------------------------------
RUN useradd --system --uid 10001 --create-home --home-dir /app appuser

WORKDIR /app

# ----------------------------------------------------------------------------
# 4. Python Dependencies (Single Source of Truth)
# ----------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ----------------------------------------------------------------------------
# 5. Application Code
# ----------------------------------------------------------------------------
COPY . .

# ----------------------------------------------------------------------------
# 6. Persistent Memory Directory (ChromaDB)
# ----------------------------------------------------------------------------
RUN mkdir -p /app/vanna_memory \
    && chown -R appuser:appuser /app

USER appuser

# ----------------------------------------------------------------------------
# 7. Exposed Ports
# ----------------------------------------------------------------------------
EXPOSE 8000 8501

# ----------------------------------------------------------------------------
# 8. Default Command (Overridden by docker-compose)
# ----------------------------------------------------------------------------
CMD ["bash"]
