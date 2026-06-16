from typing import List

import numpy as np
from loguru import logger

from .base import BaseVADBackend, VADBackendError, VADResult, VADSegment


class FsmnVADBackend(BaseVADBackend):
    name = "fsmn"

    def __init__(self, model_id: str = "fsmn-vad", model_revision: str = "v2.0.4"):
        self._model_id = model_id
        self._model_revision = model_revision
        self._model = None
        self._loaded = False

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise VADBackendError(
                "funasr is required for FsmnVADBackend. Install via `uv add funasr`."
            ) from exc

        logger.info(f"Loading FSMN-VAD model '{self._model_id}' on {device}")
        self._model = AutoModel(
            model=self._model_id,
            model_revision=self._model_revision,
            device=device,
            disable_update=True,
        )
        self._loaded = True

    def detect(self, audio: np.ndarray, sample_rate: int) -> VADResult:
        if not self._loaded:
            raise VADBackendError("FsmnVADBackend not loaded. Call load() first.")

        result = self._model.generate(input=np.asarray(audio, dtype=np.float32))
        spans = result[0].get("value", []) if result else []
        segments: List[VADSegment] = [
            VADSegment(float(beg) / 1000.0, float(end) / 1000.0) for beg, end in spans
        ]
        total = len(audio) / sample_rate if sample_rate > 0 else 0.0
        speech = sum(seg.duration for seg in segments)
        ratio = speech / total if total > 0 else 0.0
        return VADResult(segments=segments, speech_ratio=ratio, backend=self.name)

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
