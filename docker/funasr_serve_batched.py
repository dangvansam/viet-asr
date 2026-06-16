#!/usr/bin/env python3
"""Fun-ASR-Nano vLLM server with request-coalescing batcher.

Drop-in replacement for FunASR's serve_vllm.py. The pipeline sends many short
pre-cut clips concurrently; this server merges the in-flight clips into a single
vllm.LLM.generate([...]) call (the guide's batch path) instead of serializing
one-clip-per-request. OpenAI /v1/audio/transcriptions response is unchanged.
"""

import argparse
import asyncio
import io
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("funasr_batched")


def truncate_repetition(text: str, min_repeat_len: int = 3, max_repeats: int = 3) -> str:
    if not text or len(text) < 20:
        return text
    n = len(text)
    for length in range(min_repeat_len, min(n // max_repeats, 30)):
        for start in range(n - length * max_repeats):
            chunk = text[start:start + length]
            if text[start:start + length * max_repeats] == chunk * max_repeats:
                return text[:start + length]
    return text


class TranscriptionEngine:
    def __init__(self, args):
        self.args = args
        self.engine = None
        self.vad = None
        self.max_new_tokens = 500
        self.default_n_best = max(1, int(getattr(args, "n_best", 1)))
        self.default_nbest_temperature = float(getattr(args, "nbest_temperature", 0.3))

    def load(self) -> None:
        from funasr import AutoModel
        from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM

        logger.info(f"Loading vLLM engine: {self.args.model}")
        self.engine = FunASRNanoVLLM.from_pretrained(
            model=self.args.model,
            hub=self.args.hub,
            device=self.args.device,
            dtype=self.args.dtype,
            max_model_len=self.args.max_model_len,
            gpu_memory_utilization=self.args.gpu_memory_utilization,
        )
        logger.info(f"Loading VAD: {self.args.vad_model}")
        self.vad = AutoModel(model=self.args.vad_model, device=self.args.device, disable_update=True)
        logger.info("Engine ready (spk disabled, batched)")

    def to_mono_16k(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]
        audio_data = audio_data.astype(np.float32)
        if sr != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        return audio_data

    def vad_segments(self, audio: np.ndarray, sr: int = 16000) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        if len(audio) > sr * 1:
            vad_res = self.vad.generate(input=audio, fs=sr)
            spans = vad_res[0]["value"]
        else:
            spans = [[0, int(len(audio) * 1000 / sr)]]
        out = []
        for s in spans:
            s0 = int(s[0] * sr / 1000)
            s1 = int(s[1] * sr / 1000)
            clip = audio[s0:s1]
            if len(clip) > sr * 0.3:
                out.append((clip, (s[0], s[1])))
        return out

    def transcribe(self, audios: List[np.ndarray], language: Optional[str],
                   temperature: float = 0.0) -> List[dict]:
        gen_kwargs = {"max_new_tokens": self.max_new_tokens, "temperature": temperature}
        if temperature > 0:
            gen_kwargs["top_p"] = 0.95
        if language:
            gen_kwargs["language"] = language
        return self.engine.generate(inputs=audios, **gen_kwargs)

    def candidate_text(self, seg_results: List[dict]) -> str:
        return " ".join(truncate_repetition(r.get("text", "")) for r in seg_results)

    def assemble(self, seg_results: List[dict], seg_times: List[Tuple[int, int]], duration: float,
                 use_timestamp: bool) -> dict:
        output_segments = []
        full_text_parts = []
        for r, (start_ms, end_ms) in zip(seg_results, seg_times):
            text = truncate_repetition(r.get("text", ""))
            seg_info = {"text": text, "start": start_ms / 1000, "end": end_ms / 1000}
            if use_timestamp and "timestamps" in r:
                offset = start_ms / 1000
                seg_info["words"] = [
                    {"word": ts["token"], "start": ts["start_time"] + offset, "end": ts["end_time"] + offset}
                    for ts in r["timestamps"]
                ]
            output_segments.append(seg_info)
            full_text_parts.append(text)
        return {"text": " ".join(full_text_parts), "segments": output_segments, "duration": duration}


@dataclass
class BatchRequest:
    audio_data: np.ndarray
    sr: int
    language: Optional[str]
    use_timestamp: bool
    future: "asyncio.Future"
    n_best: int = 1
    nbest_temperature: float = 0.3


@dataclass
class PreparedRequest:
    request: BatchRequest
    seg_times: List[Tuple[int, int]]
    duration: float
    start: int
    count: int = field(default=0)


class TranscriptionBatcher:
    def __init__(self, engine: TranscriptionEngine, batch_size: int, wait_ms: int):
        self.engine = engine
        self.batch_size = max(1, batch_size)
        self.wait_s = max(0, wait_ms) / 1000.0
        self.queue: "asyncio.Queue[BatchRequest]" = asyncio.Queue()
        self.worker = None

    def start(self) -> None:
        self.worker = asyncio.create_task(self.run())

    async def submit(self, audio_data: np.ndarray, sr: int, language: Optional[str],
                     use_timestamp: bool, n_best: int = 1, nbest_temperature: float = 0.3) -> dict:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put(BatchRequest(
            audio_data, sr, language, use_timestamp, future, n_best, nbest_temperature,
        ))
        return await future

    async def collect_batch(self) -> List[BatchRequest]:
        first = await self.queue.get()
        batch = [first]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.wait_s
        while len(batch) < self.batch_size:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self.queue.get(), remaining))
            except asyncio.TimeoutError:
                break
        return batch

    async def run(self) -> None:
        while True:
            batch = await self.collect_batch()
            try:
                outputs = await asyncio.to_thread(self.process_batch, batch)
                for request, output in zip(batch, outputs):
                    if not request.future.done():
                        request.future.set_result(output)
            except Exception as exc:
                logger.error(f"batch failed: {exc}", exc_info=True)
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(exc)

    def process_batch(self, batch: List[BatchRequest]) -> List[dict]:
        prepared: List[PreparedRequest] = []
        flat_audios: List[np.ndarray] = []
        seg_language: List[Optional[str]] = []
        for request in batch:
            audio = self.engine.to_mono_16k(request.audio_data, request.sr)
            segs = self.engine.vad_segments(audio, 16000)
            start = len(flat_audios)
            flat_audios.extend([clip for clip, _ in segs])
            seg_language.extend([request.language] * len(segs))
            prepared.append(PreparedRequest(
                request=request,
                seg_times=[t for _, t in segs],
                duration=len(audio) / 16000,
                start=start,
                count=len(segs),
            ))

        n_best = max((r.n_best for r in batch), default=1)
        temperature = batch[0].nbest_temperature
        passes = [self.run_pass(flat_audios, seg_language, 0.0 if k == 0 else temperature)
                  for k in range(max(1, n_best))]

        outputs = []
        for item in prepared:
            span = range(item.start, item.start + item.count)
            seg_results = [passes[0][i] for i in span]
            result = self.engine.assemble(
                seg_results, item.seg_times, item.duration, item.request.use_timestamp,
            )
            if item.request.n_best > 1:
                result["candidates"] = [
                    self.engine.candidate_text([passes[k][i] for i in span])
                    for k in range(item.request.n_best)
                ]
            outputs.append(result)
        return outputs

    def run_pass(self, flat_audios: List[np.ndarray], seg_language: List[Optional[str]],
                 temperature: float) -> List[Optional[dict]]:
        results: List[Optional[dict]] = [None] * len(flat_audios)
        groups: dict = {}
        for i, language in enumerate(seg_language):
            groups.setdefault(language, []).append(i)
        for language, indices in groups.items():
            audios = [flat_audios[i] for i in indices]
            out = self.engine.transcribe(audios, language, temperature)
            for i, r in zip(indices, out):
                results[i] = r
        return results


engine: Optional[TranscriptionEngine] = None
batcher: Optional[TranscriptionBatcher] = None
app = FastAPI(title="Fun-ASR-Nano vLLM Batched Server", version="2.0")


@app.on_event("startup")
async def on_startup():
    engine.load()
    batcher.start()


@app.get("/health")
async def health():
    return {"status": "ok" if engine.engine is not None else "loading"}


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default="fun-asr-nano"),
    language: str = Form(default=None),
    response_format: str = Form(default="json"),
    timestamp_granularities: str = Form(default="word"),
    spk: bool = Form(default=False),
    n_best: int = Form(default=None),
    nbest_temperature: float = Form(default=None),
):
    content = await file.read()
    audio_data, sr = sf.read(io.BytesIO(content))
    use_ts = "word" in timestamp_granularities or "segment" in timestamp_granularities
    resolved_n = n_best if n_best is not None else engine.default_n_best
    resolved_temp = nbest_temperature if nbest_temperature is not None else engine.default_nbest_temperature
    result = await batcher.submit(audio_data, sr, language, use_ts, resolved_n, resolved_temp)
    candidates = [{"text": c} for c in result.get("candidates", [])]

    if response_format == "text":
        return JSONResponse(content=result["text"])
    if response_format == "verbose_json":
        body = {
            "task": "transcribe",
            "language": language or "zh",
            "duration": result["duration"],
            "text": result["text"],
            "segments": [
                {
                    "id": i,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "words": seg.get("words", []),
                }
                for i, seg in enumerate(result["segments"])
            ],
        }
        if candidates:
            body["candidates"] = candidates
        return JSONResponse(content=body)
    body = {"text": result["text"]}
    if candidates:
        body["candidates"] = candidates
    return JSONResponse(content=body)


@app.post("/asr")
async def asr_endpoint(
    file: UploadFile = File(...),
    language: str = Form(default=None),
    hotwords: str = Form(default=""),
    spk: bool = Form(default=False),
    timestamp: bool = Form(default=True),
):
    content = await file.read()
    audio_data, sr = sf.read(io.BytesIO(content))
    result = await batcher.submit(audio_data, sr, language, timestamp)
    return JSONResponse(content=result)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fun-ASR-Nano vLLM Batched Server")
    parser.add_argument("--port", type=int, default=9102)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model", type=str, default="FunAudioLLM/Fun-ASR-MLT-Nano-2512")
    parser.add_argument("--hub", type=str, default="ms")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--vad-model", type=str, default="fsmn-vad")
    parser.add_argument("--spk-model", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("FUNASR_BATCH_SIZE", "8")))
    parser.add_argument("--batch-wait-ms", type=int, default=int(os.environ.get("FUNASR_BATCH_WAIT_MS", "20")))
    parser.add_argument("--n-best", type=int, default=int(os.environ.get("FUNASR_NBEST", "1")))
    parser.add_argument("--nbest-temperature", type=float, default=float(os.environ.get("FUNASR_NBEST_TEMP", "0.3")))
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    engine = TranscriptionEngine(args)
    batcher = TranscriptionBatcher(engine, args.batch_size, args.batch_wait_ms)
    uvicorn.run(app, host=args.host, port=args.port)
