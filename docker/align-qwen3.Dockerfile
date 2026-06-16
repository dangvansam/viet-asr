# Isolated image for the Qwen3 forced aligner / Qwen3ASRModel. qwen-asr downgrades
# transformers (5.3 -> 4.57) and would break NeMo, so it MUST live in its own
# container. cu128 base → torch 2.10 carries sm_86 (4090) + sm_120 (Blackwell), so
# the qwen models run on either GPU. serve_qwen3_aligner.py is self-contained
# (Qwen3-ASR now serves via vLLM, so there is no serve_qwen3_asr.py).
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 HF_HOME=/root/.cache/huggingface DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir qwen-asr litserve loguru fastapi "uvicorn[standard]" python-multipart
# Restore the cu128 torch stack (qwen-asr may pull a CPU/cu124 torch) so the image
# keeps the Blackwell (sm_120) + Ada (sm_86) kernels.
RUN pip install --no-cache-dir torch==2.10.0 torchaudio==2.10.0 torchvision==0.25.0 \
        --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app
COPY scripts/serve_qwen3_aligner.py ./scripts/serve_qwen3_aligner.py

EXPOSE 8103
