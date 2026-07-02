# =============================================================================
# Multi-Agent Research Assistant — Dockerfile
# =============================================================================
# Multi-stage build: separate dependency installation from the final runtime
# image to keep layers cacheable and the final image lean.
#
# Build:   docker build -t research-assistant .
# Run:     docker run --env-file .env research-assistant
# Custom:  docker run --env-file .env research-assistant "Your query here"
# =============================================================================

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install only what's needed to build wheels
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Vasu Agrawal"
LABEL description="Multi-Agent Research Assistant — LangGraph + Groq"
LABEL org.opencontainers.image.source="https://github.com/vasu356/research-assistant"

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install pre-built wheels (no compiler, no internet during build)
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* \
    && rm -rf /wheels

# Copy application source
COPY --chown=appuser:appuser . .

# Create output directory with correct ownership
RUN mkdir -p research_outputs && chown appuser:appuser research_outputs

# Switch to non-root user
USER appuser

# Default output directory inside container
ENV OUTPUT_DIR=/app/research_outputs

# Expose the output directory as a volume so results persist on the host
VOLUME ["/app/research_outputs"]

ENTRYPOINT ["python", "main.py"]
CMD []
