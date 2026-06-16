#!/usr/bin/env python3
"""
LitServe OpenAI-compatible ASR service. Hosts one in-process ASR backend
(nemotron/vietasr/funasr) behind `POST /v1/audio/transcriptions` with LitServe
dynamic batching, so the pipeline talks to it via the unified `openai_transcription`
client — no torch/nemo/funasr in the pipeline venv. Engines that natively serve the
OpenAI transcription API (Qwen3-ASR / Fun-ASR via vLLM) are used directly.

Backends exposing `transcribe_batch` (e.g. nemotron NeMo) get true GPU batching;
others (e.g. vietasr) loop single `transcribe` calls per worker and scale via
`--workers`. Both share `TranscriptionLitAPI`.

Contract (see serving/lit_api.TranscriptionLitAPI):
    POST /v1/audio/transcriptions  multipart: file=<wav>, model, language,
        response_format, timestamp_granularities[], include[], prompt
    -> verbose_json {task, language, duration, text, segments[...], words[...]}

Run (each backend in its own env/container):
    python scripts/serve_asr.py --backend nemotron --device cuda --port 9103 \
        --kw model_name=nvidia/nemotron-3.5-asr-streaming-0.6b --kw target_lang=vi-VN
    python scripts/serve_asr.py --backend vietasr --device cuda --port 9104 --workers 2
"""

import argparse
import os

from multitalker_asr.serving import TranscriptionLitAPI, run_lit_service


def parse_kw(pairs):
    kwargs = {}
    for item in pairs or []:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        kwargs[key] = value
    return kwargs


def main():
    p = argparse.ArgumentParser(description="Serve an ASR backend (OpenAI transcription API, LitServe)")
    p.add_argument("--backend", default="nemotron")
    p.add_argument("--device", default="cpu")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9103)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--language", default=None)
    p.add_argument("--model-id", default=None, help="model id echoed in responses / /v1/models")
    p.add_argument("--kw", action="append", default=[], help="backend kwarg key=value (repeatable)")
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "8")))
    p.add_argument("--batch-wait-ms", type=int, default=int(os.environ.get("BATCH_WAIT_MS", "20")))
    p.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "1")))
    args = p.parse_args()

    kwargs = parse_kw(args.kw)
    model_id = args.model_id or kwargs.get("model_name") or kwargs.get("model_id") or args.backend

    api = TranscriptionLitAPI(
        backend_name=args.backend,
        backend_kwargs=kwargs,
        sample_rate=args.sample_rate,
        language=args.language,
        device=args.device,
        model_id=model_id,
        max_batch_size=args.batch_size,
        batch_wait_ms=args.batch_wait_ms,
    )
    accelerator = "cpu" if args.device == "cpu" else ("cuda" if "cuda" in args.device else "auto")
    run_lit_service(
        api, args.host, args.port,
        accelerator=accelerator, workers_per_device=args.workers, model_id=model_id,
    )


if __name__ == "__main__":
    main()
