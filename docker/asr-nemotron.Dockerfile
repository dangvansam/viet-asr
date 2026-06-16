# GPU image for NeMo-based services (nemotron streaming ASR). Base ships python
# 3.11 + CUDA torch, so we avoid the ubuntu-3.10 mismatch and only add NeMo.
# nemotron-3.5 needs a specific NeMo commit overlay (not the pinned main) — see
# project memory project_python311_nemotron_overlay. Build arg NEMO_REV applies it.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG NEMO_REV=160a7428
ENV PYTHONUNBUFFERED=1 HF_HOME=/root/.cache/huggingface DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        loguru soundfile librosa requests omegaconf \
        litserve fastapi "uvicorn[standard]" python-multipart
RUN pip install --no-cache-dir \
        "nemo-toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@${NEMO_REV}"
# NeMo@160a7428 pulls torch 2.12.0+cu130, orphaning the base torchaudio/torchvision
# (libtorchaudio undefined-symbol / torchvision::nms errors). Reinstall both from
# the matching cu130 index so the stack is consistent. cu130 also enables Blackwell.
RUN pip install --no-cache-dir torchaudio torchvision \
        --index-url https://download.pytorch.org/whl/cu130

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-cache-dir --no-deps .

EXPOSE 9103
