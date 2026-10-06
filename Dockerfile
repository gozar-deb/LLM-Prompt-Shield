# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/app

# System deps needed to build a couple of wheels (cryptography-adjacent
# transitive deps, etc). Kept minimal on purpose.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# Add `--build-arg INSTALL_ML=1` (and uncomment below) if you want the ONNX
# classifier baked into the image. By default the image ships with the
# lightweight heuristic backend only — see requirements-ml.txt.
ARG INSTALL_ML=0
COPY requirements-ml.txt ./
RUN pip install -r requirements.txt \
    && if [ "$INSTALL_ML" = "1" ]; then pip install -r requirements-ml.txt; fi

COPY app ./app
COPY scripts ./scripts

# Non-root user
RUN useradd --create-home --uid 1000 shield \
    && mkdir -p /srv/app/logs \
    && chown -R shield:shield /srv/app
USER shield

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# 4 workers is a reasonable default for a CPU-bound classifier; tune to
# your core count. Use --workers 1 if you enable the ONNX backend on a GPU.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
