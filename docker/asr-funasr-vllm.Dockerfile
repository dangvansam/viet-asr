# Fun-ASR-MLT-Nano served by vLLM via modelscope/FunASR `serve_vllm.py`, which
# exposes the OpenAI transcription API (POST /v1/audio/transcriptions, verbose_json
# with word timestamps + /asr diarization). The pipeline's unified
# `openai_transcription` client talks to it — same contract as asr-qwen3 / nemotron.
#
# Mirrors the proven host env (pytorch 2.10+cu128 dual-arch sm_86+sm_120, funasr +
# vllm + modelscope, FLASH_ATTN on Blackwell, ModelScope download since HF LFS
# stalls). Model files persist via MODELSCOPE_CACHE on the mounted volume.
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 HF_HOME=/root/.cache/huggingface DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    MODELSCOPE_CACHE=/root/.cache/huggingface/modelscope

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        "funasr>=1.3.0" "vllm>=0.12.0" modelscope \
        safetensors tiktoken websockets regex fastapi "uvicorn[standard]" \
        python-multipart loguru soundfile librosa requests
# vllm>=0.12 ships its own cu13 nvidia wheels (libcudart.so.13) whose lib dirs aren't
# on the loader path → `vllm._C` ImportError. Register every nvidia/*/lib via ldconfig
# (covers both the base cu12 and vllm's cu13 runtimes).
RUN find /usr/local/lib/python3.12/dist-packages/nvidia -type d -name lib 2>/dev/null \
        > /etc/ld.so.conf.d/nvidia-cu.conf \
    && find /opt/conda/lib/python*/site-packages/nvidia -type d -name lib 2>/dev/null \
        >> /etc/ld.so.conf.d/nvidia-cu.conf || true; ldconfig

# modelscope/FunASR ships the offline vLLM service script (serve_vllm.py).
RUN git clone --depth 1 https://github.com/modelscope/FunASR.git /opt/FunASR \
    && pip install --no-cache-dir --no-deps -e /opt/FunASR

WORKDIR /opt/FunASR/examples/industrial_data_pretraining/fun_asr_nano
COPY docker/funasr_serve_batched.py /opt/FunASR/examples/industrial_data_pretraining/fun_asr_nano/serve_vllm.py
EXPOSE 9102
