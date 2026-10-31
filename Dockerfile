
# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:$PATH" \
    UV_HTTP_TIMEOUT=120 \
    UV_HTTP_RETRIES=3 \
    UV_CONCURRENT_DOWNLOADS=10 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Install dependencies separately so Docker can cache this layer.
COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy application files only after dependencies are installed.
COPY src ./src
COPY frontend ./frontend
COPY artifacts ./artifacts
COPY reports ./reports

# Install the project itself, reusing the existing environment.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    --start-period=20s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"]

CMD ["uvicorn", "sms_spam_mlops.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
