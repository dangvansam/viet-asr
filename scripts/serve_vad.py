#!/usr/bin/env python3
"""
LitServe VAD service. Hosts one VAD backend (silero/ten/pyannote_seg/fsmn/consensus)
behind `POST /v1/audio/vad` with LitServe dynamic batching, so the data pipeline can
run VAD via URL only — no in-process silero/ten/pyannote in the pipeline venv.

Contract (consumed by ServiceVADBackend) — shared OpenAI-style envelope:
    POST /v1/audio/vad   multipart: file=<wav>
    -> {"task": "vad", "segments": [{"id": int, "start": float, "end": float}],
        "speech_ratio": float, "backend": str, "raw": {...}}

Run:
    uv run --extra vad python scripts/serve_vad.py --backend silero --port 9001
    # consensus needs the providers installed; HF_TOKEN for pyannote_seg:
    uv run --extra vad python scripts/serve_vad.py --backend consensus \
        --providers silero,pyannote_seg,ten --strategy majority --port 9001
"""

import argparse
import os

from multitalker_asr.serving import VadLitAPI, run_lit_service


def main():
    p = argparse.ArgumentParser(description="Serve a VAD backend over HTTP (LitServe)")
    p.add_argument("--backend", default="silero")
    p.add_argument("--device", default="cpu")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9001)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--providers", default="", help="consensus: comma-separated backend names")
    p.add_argument("--strategy", default="majority", help="consensus: majority|intersection|union")
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "1")))
    p.add_argument("--batch-wait-ms", type=int, default=int(os.environ.get("BATCH_WAIT_MS", "0")))
    p.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "1")))
    args = p.parse_args()

    kwargs = {}
    if args.backend in ("pyannote_seg", "consensus") and args.hf_token:
        kwargs["hf_token"] = args.hf_token
    if args.backend == "consensus":
        kwargs["strategy"] = args.strategy
        if args.providers:
            kwargs["providers"] = [{"name": n} for n in args.providers.split(",")]

    api = VadLitAPI(
        backend_name=args.backend,
        backend_kwargs=kwargs,
        sample_rate=args.sample_rate,
        device=args.device,
        max_batch_size=args.batch_size,
        batch_wait_ms=args.batch_wait_ms,
    )
    accelerator = "cpu" if args.device == "cpu" else ("cuda" if "cuda" in args.device else "auto")
    run_lit_service(
        api, args.host, args.port,
        accelerator=accelerator, workers_per_device=args.workers, model_id=args.backend,
    )


if __name__ == "__main__":
    main()
