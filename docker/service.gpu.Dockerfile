# Dual-GPU service image for VAD / mms_fa align / funasr / vietasr. Base ships
# torch+torchaudio+torchvision 2.10+cu128 (the project's pin) which carries BOTH
# sm_86 (→ runs on the RTX 4090 / GPU 0 by binary compat) AND sm_120 (→ runs on the
# Blackwell / GPU 1), so every service can be placed on either GPU. Installs a
# curated set of model libs (EXTRA_PIP) plus the project package without heavy deps.
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

ARG EXTRA_PIP=""
# the cu128 base's python is PEP-668 externally-managed → allow pip to install.
ENV PYTHONUNBUFFERED=1 HF_HOME=/root/.cache/huggingface DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1

# libc++1 for ten-vad; ffmpeg/libsndfile for audio I/O.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libc++1 libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        loguru soundfile librosa requests omegaconf \
        litserve fastapi "uvicorn[standard]" python-multipart
RUN if [ -n "${EXTRA_PIP}" ]; then pip install --no-cache-dir ${EXTRA_PIP}; fi
# Restore the cu128 torch stack in case a model lib downgraded it (would drop the
# sm_120 Blackwell kernels). No-op when already satisfied.
RUN pip install --no-cache-dir torch==2.10.0 torchaudio==2.10.0 torchvision==0.25.0 \
        --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-cache-dir --no-deps .

EXPOSE 9001 9102 9104 9203
