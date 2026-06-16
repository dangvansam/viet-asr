# Generic CPU/GPU-light service image for VAD / mms_fa align / funasr / vietasr.
# Installs a curated set of model libs (BASE_PIP + EXTRA_PIP) plus the project
# package WITHOUT its heavy core deps (no git NeMo) — each service only pulls what
# it actually needs, keeping images small and env-isolated.
FROM python:3.11-slim

ARG BASE_PIP="loguru numpy soundfile requests omegaconf fastapi uvicorn[standard] python-multipart torch --index-url https://download.pytorch.org/whl/cpu"
ARG EXTRA_PIP=""

ENV PYTHONUNBUFFERED=1 HF_HOME=/root/.cache/huggingface

# libc++1 is required by ten-vad; ffmpeg/libsndfile for audio I/O.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libc++1 libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir ${BASE_PIP}
RUN if [ -n "${EXTRA_PIP}" ]; then pip install --no-cache-dir ${EXTRA_PIP}; fi

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-cache-dir --no-deps .

EXPOSE 9001 9102 9104 9203
