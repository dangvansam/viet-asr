#!/usr/bin/env python3
"""
LitServe forced-alignment service. Hosts one in-process align backend (mms_fa/nemo_nfa)
behind `POST /v1/audio/alignments` with LitServe dynamic batching, using the same
contract as the Qwen3 aligner, so the pipeline aligns via the URL-only `service`
align client — no torchaudio/NeMo in the pipeline venv.

Contract (consumed by ServiceAlignBackend / Qwen3ServiceAlignBackend) — shared
OpenAI-style envelope:
    POST /v1/audio/alignments  multipart: file=<wav>, text=<str>, language=<str>
    -> {"task": "align", "text": str, "score": float,
        "words": [{"word": str, "start": float, "end": float}]}

Run:
    uv run python scripts/serve_align.py --backend mms_fa --device cuda --port 9203
    python scripts/serve_align.py --backend nemo_nfa --device cuda --port 9203 \
        --kw model_path=models/some_ctc.nemo
"""

import argparse
import os

from multitalker_asr.serving import AlignLitAPI, run_lit_service


def parse_kw(pairs):
    kwargs = {}
    for item in pairs or []:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        kwargs[key] = value
    return kwargs


def main():
    p = argparse.ArgumentParser(description="Serve an alignment backend over HTTP (LitServe)")
    p.add_argument("--backend", default="mms_fa")
    p.add_argument("--device", default="cuda")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9203)
    p.add_argument("--language", default="Vietnamese")
    p.add_argument("--kw", action="append", default=[], help="backend kwarg key=value (repeatable)")
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "8")))
    p.add_argument("--batch-wait-ms", type=int, default=int(os.environ.get("BATCH_WAIT_MS", "20")))
    p.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "1")))
    args = p.parse_args()

    api = AlignLitAPI(
        backend_name=args.backend,
        backend_kwargs=parse_kw(args.kw),
        language=args.language,
        device=args.device,
        model_id=args.backend,
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
