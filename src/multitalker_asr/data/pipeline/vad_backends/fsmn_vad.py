from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from .base import BaseVADBackend, VADBackendError, VADResult, VADSegment


class FsmnVADBackend(BaseVADBackend):
    name = "fsmn"

    def __init__(
        self,
        model_id: str = "fsmn-vad",
        model_revision: str = "v2.0.4",
        dynamic_silence: Optional[bool] = None,
        silence_schedule: Optional[List[Tuple[float, float]]] = None,
        speech_noise_thres: Optional[float] = None,
        max_end_silence_time: Optional[int] = None,
    ):
        self._model_id = model_id
        self._model_revision = model_revision
        self._dynamic_silence = dynamic_silence
        self._silence_schedule = silence_schedule
        self._speech_noise_thres = speech_noise_thres
        self._max_end_silence_time = max_end_silence_time
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

        gen_kwargs = {}
        if self._dynamic_silence is not None:
            gen_kwargs["dynamic_silence"] = self._dynamic_silence
        if self._silence_schedule is not None:
            gen_kwargs["silence_schedule"] = self._silence_schedule
        if self._speech_noise_thres is not None:
            gen_kwargs["speech_noise_thres"] = self._speech_noise_thres
        if self._max_end_silence_time is not None:
            gen_kwargs["max_end_silence_time"] = self._max_end_silence_time

        result = self._model.generate(
            input=np.asarray(audio, dtype=np.float32), **gen_kwargs
        )
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
