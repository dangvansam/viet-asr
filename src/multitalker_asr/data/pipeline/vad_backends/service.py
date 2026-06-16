import io
from typing import List

import numpy as np
from loguru import logger

from .base import BaseVADBackend, VADBackendError, VADResult, VADSegment


class ServiceVADBackend(BaseVADBackend):
    """URL-only VAD client: POSTs audio to a VAD service (scripts/serve_vad.py)
    and parses the returned speech segments. Keeps the pipeline venv free of any
    in-process VAD model (silero/ten/pyannote)."""

    name = "service"

    def __init__(self, base_url: str = "http://127.0.0.1:9001", timeout: float = 120.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = None
        self._loaded = False

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        try:
            import requests
        except ImportError as exc:
            raise VADBackendError("requests is required for ServiceVADBackend.") from exc
        self._session = requests.Session()
        self._loaded = True
        logger.info(f"ServiceVADBackend ready (base_url={self._base_url})")

    def detect(self, audio: np.ndarray, sample_rate: int) -> VADResult:
        if not self._loaded:
            raise VADBackendError("ServiceVADBackend not loaded. Call load() first.")

        wav_bytes = self._to_wav_bytes(audio, sample_rate)
        try:
            response = self._session.post(
                f"{self._base_url}/v1/audio/vad",
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise VADBackendError(f"ServiceVADBackend request failed: {exc}") from exc

        segments: List[VADSegment] = [
            VADSegment(float(s["start"]), float(s["end"]))
            for s in (payload.get("segments") or [])
        ]
        return VADResult(
            segments=segments,
            speech_ratio=float(payload.get("speech_ratio", 0.0)),
            backend=payload.get("backend", self.name),
            raw=payload.get("raw") or {},
        )

    def _to_wav_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise VADBackendError("soundfile is required for ServiceVADBackend.") from exc
        buf = io.BytesIO()
        sf.write(buf, np.asarray(audio, dtype=np.float32), sample_rate, format="WAV")
        return buf.getvalue()

    def unload(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        self._loaded = False
