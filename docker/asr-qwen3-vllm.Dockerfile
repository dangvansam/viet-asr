# Qwen3-ASR served by vLLM (OpenAI-compatible /v1/chat/completions with audio),
# per https://github.com/QwenLM/Qwen3-ASR#deployment-with-vllm. vLLM nightly cu129
# supports Blackwell (sm_120) → runs on GPU 1. The pipeline's `qwen3_vllm` client
# talks to it. Kept isolated (its own torch cu129 stack).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 HF_HOME=/root/.cache/huggingface DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# vLLM nightly (cu129) + audio extra, exactly the Qwen3-ASR guide's indexes.
RUN uv pip install --system -U "vllm[audio]" --pre \
        --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        --index-strategy unsafe-best-match

# The nightly mixes cu12/cu13 nvidia wheels; their .so dirs aren't on the loader
# path (libcudart.so.13 not found). Register every nvidia/*/lib via ldconfig.
RUN find /usr/local/lib/python3.12/site-packages/nvidia -type d -name lib \
        > /etc/ld.so.conf.d/nvidia-cu.conf && ldconfig

# vLLM's flashinfer compiles CUDA kernels at runtime (sm_120 on Blackwell) → needs
# a C compiler AND the full CUDA headers. The cu13 toolkit (CUDA_HOME) lacks curand
# etc. (split across nvidia/* wheels), so consolidate every nvidia header + lib into
# the cu13 toolkit dir that flashinfer's nvcc points at.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN CU=/usr/local/lib/python3.12/site-packages/nvidia/cu13; \
    for d in /usr/local/lib/python3.12/site-packages/nvidia/*/include; do \
        [ "$d" = "$CU/include" ] || cp -rn "$d"/. "$CU/include/" 2>/dev/null || true; \
    done; \
    for d in /usr/local/lib/python3.12/site-packages/nvidia/*/lib; do \
        [ "$d" = "$CU/lib" ] || cp -rn "$d"/. "$CU/lib/" 2>/dev/null || true; \
    done; \
    cd "$CU/lib" && for f in *.so.*; do \
        base=$(echo "$f" | sed 's/\.so\..*/.so/'); [ -e "$base" ] || ln -s "$f" "$base"; \
    done; ldconfig

ENV CUDA_HOME=/usr/local/lib/python3.12/site-packages/nvidia/cu13 \
    LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib
EXPOSE 8101
