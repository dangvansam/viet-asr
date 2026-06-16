#!/usr/bin/env python3
"""
LitServe wrapper for Qwen3-ForcedAligner — run in a SEPARATE venv (qwen-asr downgrades
transformers 5.3→4.57 and breaks NeMo, so it must NOT live in the project venv).

Self-contained on purpose: the align-qwen3 image only copies this one file (it does
NOT install the multitalker_asr package), so the LitAPI is implemented inline rather
than importing the shared base. The in-venv `qwen3_service` align client POSTs
(wav + text + language) here over the same OpenAI-style contract.

Setup (separate venv — also needs litserve):
    uv venv /tmp/qwen-aligner && source /tmp/qwen-aligner/bin/activate
    uv pip install qwen-asr litserve loguru fastapi "uvicorn[standard]" python-multipart
    CUDA_VISIBLE_DEVICES=1 python scripts/serve_qwen3_aligner.py \
        --model Qwen/Qwen3-ForcedAligner-0.6B --host 127.0.0.1 --port 8103

Contract:
    POST /v1/audio/alignments  multipart: file=<wav>, text=<str>, language=<str>
    -> {"task": "align", "language": str, "text": str,
        "words": [{"word": str, "start": float, "end": float}, ...]}
"""

import argparse
import os
import tempfile
import time

import litserve as ls
from loguru import logger


class QwenAlignerLitAPI(ls.LitAPI):
    """Hosts Qwen3-ForcedAligner with LitServe dynamic batching; batches a clip's
    (audio, text, language) triples into one `aligner.align(...)` call."""

    def __init__(self, model, dtype, device, max_batch_size, batch_timeout):
        super().__init__(
            max_batch_size=max_batch_size,
            batch_timeout=batch_timeout,
            api_path="/v1/audio/alignments",
        )
        self._model = model
        self._dtype = dtype
        self._device = device
        self._aligner = None

    def setup(self, device):
        import torch
        from qwen_asr import Qwen3ForcedAligner

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dev = self._device or device
        device_map = f"{dev}:0" if "cuda" in str(dev) else str(dev)
        self._aligner = Qwen3ForcedAligner.from_pretrained(
            self._model, dtype=dtype_map.get(self._dtype, torch.bfloat16), device_map=device_map
        )
        logger.info(f"Qwen3-ForcedAligner ready model={self._model} device_map={device_map}")

    def decode_request(self, request):
        upload = request.get("file") if hasattr(request, "get") else None
        if upload is None or not hasattr(upload, "file"):
            raise ValueError("missing multipart file field 'file'")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            tmp.write(upload.file.read())
            tmp.flush()
        finally:
            tmp.close()
        return {
            "path": tmp.name,
            "text": request.get("text", ""),
            "language": request.get("language") or "Vietnamese",
            "t0": time.perf_counter(),
        }

    def predict(self, batch):
        audio = [it["path"] for it in batch]
        texts = [it["text"] for it in batch]
        langs = [it["language"] for it in batch]
        try:
            results = self._aligner.align(audio=audio, text=texts, language=langs)
        except Exception as exc:
            logger.warning(f"qwen align failed: {exc}")
            results = [None] * len(batch)
        finally:
            for path in audio:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        outputs = []
        for it, res in zip(batch, results):
            words = [
                {"word": w.text, "start": float(w.start_time), "end": float(w.end_time)}
                for w in (res or [])
            ]
            outputs.append({
                "language": it["language"],
                "text": it["text"],
                "words": words,
                "elapsed_s": round(time.perf_counter() - it["t0"], 4),
            })
        return outputs

    def encode_response(self, output):
        return {"task": "align", **output}


def main():
    p = argparse.ArgumentParser(description="Serve Qwen3-ForcedAligner over HTTP (LitServe)")
    p.add_argument("--model", default="Qwen/Qwen3-ForcedAligner-0.6B")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8103)
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "4")))
    p.add_argument("--batch-wait-ms", type=int, default=int(os.environ.get("BATCH_WAIT_MS", "20")))
    args = p.parse_args()

    api = QwenAlignerLitAPI(
        model=args.model,
        dtype=args.dtype,
        device=args.device,
        max_batch_size=args.batch_size,
        batch_timeout=max(args.batch_wait_ms, 0) / 1000.0,
    )
    accelerator = "cpu" if args.device == "cpu" else ("cuda" if "cuda" in args.device else "auto")
    server = ls.LitServer(api, accelerator=accelerator, devices=1)

    @server.app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": args.model, "object": "model"}]}

    logger.info(f"Serving Qwen3-ForcedAligner on http://{args.host}:{args.port}")
    server.run(host=args.host, port=args.port, generate_client_file=False)


if __name__ == "__main__":
    main()
