"""
LitServe-based audio model micro-services (VAD, ASR, alignment).

Replaces the hand-rolled FastAPI factory + ServiceBatcher with LitServe's native
dynamic batching while preserving the exact OpenAI Audio API contract the pipeline
clients depend on (`POST /v1/audio/{vad,transcriptions,alignments}`, multipart
`file=<wav>` + OpenAI form fields, the shared response envelope).

How the contract maps onto LitServe (validated against litserve 0.2.17):
- `decode_request(self, request)` is left UNANNOTATED, so LitServe hands it the
  Starlette `FormData` for a multipart body. The upload bytes are read from the
  sync `request["file"].file` (the async `UploadFile.read()` cannot be awaited in
  the worker), and the OpenAI params come from `parse_openai_form` unchanged.
- `predict(self, batch)` receives `List[item]` (LitServe's default batch/unbatch
  pass plain lists straight through), runs the backend's coalesced inference, and
  returns one result per item — the old `build_batch_*` bodies port in directly.
- `encode_response(self, item)` builds the per-request envelope; the original
  request params are carried through `predict` because LitServe does not expose the
  request at encode time.

The model is loaded in `setup(device)` (run once per worker process), so each
LitAPI stores only picklable config — never the loaded backend — to survive being
shipped to the worker.
"""

import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import litserve as ls
from loguru import logger

from .app_factory import asr_result_to_verbose_json, parse_openai_form


def normalize_device(device: str) -> str:
    """Collapse a LitServe device string (`cuda:0`, `cpu`) to the `cpu`/`cuda` form
    the in-venv backends expect; specific GPU selection is done via CUDA_VISIBLE_DEVICES."""
    if device is None:
        return "cpu"
    text = str(device)
    if text == "cpu" or text.startswith("cpu"):
        return "cpu"
    if text.startswith("cuda"):
        return "cuda"
    return text


class BaseAudioLitAPI(ls.LitAPI):
    """Shared multipart decode + temp-WAV + envelope plumbing for audio LitAPIs.

    Subclasses implement `setup(device)` (load the backend) and `infer(items)`
    (batched inference over `(audio_path, params)` pairs returning one output per
    item); `envelope(output, params)` shapes the per-request response payload.
    """

    task: str = ""

    def __init__(
        self,
        model_id: str = "",
        file_field: str = "file",
        max_batch_size: int = 1,
        batch_wait_ms: int = 0,
        api_path: str = "/predict",
        suffix: str = ".wav",
    ):
        super().__init__(
            max_batch_size=max_batch_size,
            batch_timeout=max(batch_wait_ms, 0) / 1000.0,
            api_path=api_path,
        )
        self._model_id = model_id
        self._file_field = file_field
        self._suffix = suffix

    def decode_request(self, request) -> Dict[str, Any]:
        upload = request.get(self._file_field) if hasattr(request, "get") else None
        if upload is None or not hasattr(upload, "file"):
            raise ValueError(f"missing multipart file field '{self._file_field}'")
        audio_bytes = upload.file.read()
        params = parse_openai_form(request)
        tmp = tempfile.NamedTemporaryFile(suffix=self._suffix, delete=False)
        try:
            tmp.write(audio_bytes)
            tmp.flush()
        finally:
            tmp.close()
        return {"path": tmp.name, "params": params, "t0": time.perf_counter()}

    def predict(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        items: List[Tuple[str, Dict[str, Any]]] = [(it["path"], it["params"]) for it in batch]
        try:
            outputs = self.infer(items)
        finally:
            for it in batch:
                self._cleanup(it["path"])
        results: List[Dict[str, Any]] = []
        for it, output in zip(batch, outputs):
            results.append(
                {
                    "output": output,
                    "params": it["params"],
                    "elapsed": round(time.perf_counter() - it["t0"], 4),
                }
            )
        return results

    def encode_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        payload = self.envelope(response["output"], response["params"])
        if self.task:
            payload.setdefault("task", self.task)
        payload["elapsed_s"] = response["elapsed"]
        return payload

    def infer(self, items: List[Tuple[str, Dict[str, Any]]]) -> List[Any]:
        raise NotImplementedError

    def envelope(self, output: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def _cleanup(path: str) -> None:
        try:
            os.unlink(path)
        except OSError:
            pass


class VadLitAPI(BaseAudioLitAPI):
    """`POST /v1/audio/vad` — hosts one VAD backend (silero/ten/pyannote_seg/consensus)."""

    task = "vad"

    def __init__(self, backend_name: str, backend_kwargs: Dict[str, Any],
                 sample_rate: int = 16000, device: str = "cpu", **kwargs):
        kwargs.setdefault("api_path", "/v1/audio/vad")
        kwargs.setdefault("model_id", backend_name)
        super().__init__(**kwargs)
        self._backend_name = backend_name
        self._backend_kwargs = backend_kwargs
        self._sample_rate = sample_rate
        self._device = device
        self._backend = None
        self._loader = None

    def setup(self, device):
        from ..data.pipeline.vad_backends import build_vad_backend
        from ..utils.audio import AudioLoader

        self._backend = build_vad_backend(self._backend_name, **self._backend_kwargs)
        self._backend.load(device=normalize_device(self._device or device))
        self._loader = AudioLoader(target_sample_rate=self._sample_rate)
        logger.info(f"VAD backend={self._backend_name} ready device={normalize_device(self._device or device)}")

    def infer(self, items):
        outputs = []
        for path, _params in items:
            audio, sr = self._loader.load(path)
            result = self._backend.detect(audio, sr)
            outputs.append(
                {
                    "segments": [
                        {"id": i, "start": s.start, "end": s.end}
                        for i, s in enumerate(result.segments)
                    ],
                    "speech_ratio": result.speech_ratio,
                    "backend": result.backend,
                    "raw": result.raw,
                }
            )
        return outputs

    def envelope(self, output, params):
        return dict(output)


class TranscriptionLitAPI(BaseAudioLitAPI):
    """`POST /v1/audio/transcriptions` — hosts one in-venv ASR backend (nemotron/vietasr).

    Coalesces concurrent clips, groups by language, and uses `transcribe_batch`
    (true GPU batch, e.g. nemotron) when the backend exposes it, else loops single
    `transcribe` calls on the worker. Mirrors the old `build_batch_transcribe`.
    """

    task = "transcribe"

    def __init__(self, backend_name: str, backend_kwargs: Dict[str, Any],
                 sample_rate: int = 16000, language: Optional[str] = None,
                 device: str = "cpu", **kwargs):
        kwargs.setdefault("api_path", "/v1/audio/transcriptions")
        super().__init__(**kwargs)
        self._backend_name = backend_name
        self._backend_kwargs = backend_kwargs
        self._sample_rate = sample_rate
        self._language = language
        self._device = device
        self._backend = None
        self._loader = None
        self._has_batch = False

    def setup(self, device):
        from ..data.pipeline.asr_backends import build_asr_backend
        from ..utils.audio import AudioLoader

        self._backend = build_asr_backend(self._backend_name, **self._backend_kwargs)
        self._backend.load(device=normalize_device(self._device or device))
        self._loader = AudioLoader(target_sample_rate=self._sample_rate)
        self._has_batch = hasattr(self._backend, "transcribe_batch")
        logger.info(f"ASR backend={self._backend_name} ready batch={self._has_batch}")

    def infer(self, items):
        results: List[Any] = [None] * len(items)
        loaded = []
        groups: Dict[Optional[str], List[int]] = {}
        for i, (path, params) in enumerate(items):
            audio, sr = self._loader.load(path)
            loaded.append((audio, sr))
            lang = params.get("language") or self._language
            groups.setdefault(lang, []).append(i)
        for language, idxs in groups.items():
            if self._has_batch:
                audios = [loaded[i][0] for i in idxs]
                out = self._backend.transcribe_batch(audios, loaded[idxs[0]][1], language)
                for i, res in zip(idxs, out):
                    results[i] = res
            else:
                for i in idxs:
                    audio, sr = loaded[i]
                    results[i] = self._backend.transcribe(audio, sr, language)
        return results

    def envelope(self, output, params):
        granularities = params.get("timestamp_granularities") or ["segment"]
        include = params.get("include") or []
        payload = asr_result_to_verbose_json(output, granularities, include, self._model_id)
        fmt = params.get("response_format", "verbose_json")
        if fmt == "json":
            return {"text": payload["text"]}
        return payload


class AlignLitAPI(BaseAudioLitAPI):
    """`POST /v1/audio/alignments` — hosts one forced-align backend (mms_fa/nemo_nfa).

    Uses `align_batch` (mms_fa reuses the wav2vec2 emission across a clip's texts)
    when available, else loops single `align` calls. Mirrors `build_batch_align`.
    """

    task = "align"

    def __init__(self, backend_name: str, backend_kwargs: Dict[str, Any],
                 language: str = "Vietnamese", device: str = "cuda", **kwargs):
        kwargs.setdefault("api_path", "/v1/audio/alignments")
        kwargs.setdefault("model_id", backend_name)
        super().__init__(**kwargs)
        self._backend_name = backend_name
        self._backend_kwargs = backend_kwargs
        self._language = language
        self._device = device
        self._backend = None
        self._has_batch = False

    def setup(self, device):
        from ..data.pipeline.align_backends import build_align_backend

        self._backend = build_align_backend(self._backend_name, **self._backend_kwargs)
        self._backend.load(device=normalize_device(self._device or device))
        self._has_batch = hasattr(self._backend, "align_batch")
        logger.info(f"Align backend={self._backend_name} ready batch={self._has_batch}")

    def infer(self, items):
        triples = [
            (path, params.get("text", ""), params.get("language") or self._language)
            for path, params in items
        ]
        if self._has_batch:
            results = self._backend.align_batch(triples)
            return [self._payload(t[1], t[2], r) for t, r in zip(triples, results)]
        outputs = []
        for path, text, lang in triples:
            try:
                outputs.append(self._payload(text, lang, self._backend.align(path, text, lang)))
            except Exception as exc:
                logger.warning(f"align failed: {exc}")
                outputs.append({"language": lang, "text": text, "score": 0.0, "words": [], "error": str(exc)})
        return outputs

    def envelope(self, output, params):
        return dict(output)

    @staticmethod
    def _payload(text, language, result):
        words = [
            {"word": w.text, "start": w.start_time, "end": w.end_time}
            for w in result.words
        ]
        return {"language": language, "text": text, "score": result.score, "words": words}


def run_lit_service(api: BaseAudioLitAPI, host: str, port: int,
                    accelerator: str = "auto", devices=1, workers_per_device: int = 1,
                    model_id: str = "") -> None:
    """Build a LitServer for `api`, mount `/v1/models`, and serve on host:port.

    GPU pinning is left to CUDA_VISIBLE_DEVICES / CDI (devices is a worker count),
    matching the existing docker-compose placement. `workers_per_device` gives
    thread-safe CPU backends (e.g. vietasr) extra concurrency without coalescing.
    """
    server = ls.LitServer(
        api,
        accelerator=accelerator,
        devices=devices,
        workers_per_device=workers_per_device,
        model_metadata={"model": model_id} if model_id else None,
    )

    @server.app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_id, "object": "model"}]}

    logger.info(f"Serving {api.api_path} on http://{host}:{port} (accelerator={accelerator})")
    server.run(host=host, port=port, generate_client_file=False)
