# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps (ffmpeg for audio, build tools for some pip packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps into a prefix we can copy
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install \
    edge-tts \
    openai \
    fastapi \
    "uvicorn[standard]" \
    python-dotenv \
    pandas \
    pydantic

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Hindi TTS API"
LABEL org.opencontainers.image.description="Production Hindi Text-to-Speech REST API"
LABEL org.opencontainers.image.source="https://github.com/NishchayVashishtha/HindiTTS"

WORKDIR /app

# Only ffmpeg needed at runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/        ./src/
COPY api/        ./api/
COPY outputs/    ./outputs/
COPY .env.example .env

# Create outputs dir
RUN mkdir -p /app/outputs

# Non-root user for security
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
