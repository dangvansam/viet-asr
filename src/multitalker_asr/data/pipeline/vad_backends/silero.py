from typing import List

import numpy as np
from loguru import logger

from .base import BaseVADBackend, VADBackendError, VADResult, VADSegment


class SileroVADBackend(BaseVADBackend):
    name = "silero"

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        use_onnx: bool = False,
    ):
        self._threshold = threshold
        self._min_speech_ms = min_speech_duration_ms
        self._min_silence_ms = min_silence_duration_ms
        self._use_onnx = use_onnx
        self._model = None
        self._get_speech_timestamps = None
        self._loaded = False

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad
        except ImportError as exc:
            raise VADBackendError(
                "silero-vad is required for SileroVADBackend. Install via `uv add silero-vad`."
            ) from exc

        logger.info("Loading silero-vad model")
        self._model = load_silero_vad(onnx=self._use_onnx)
        self._get_speech_timestamps = get_speech_timestamps
        self._loaded = True

    def detect(self, audio: np.ndarray, sample_rate: int) -> VADResult:
        if not self._loaded:
            raise VADBackendError("SileroVADBackend not loaded. Call load() first.")

        import torch

        wav = torch.as_tensor(np.asarray(audio, dtype=np.float32))
        stamps = self._get_speech_timestamps(
            wav,
            self._model,
            sampling_rate=sample_rate,
            threshold=self._threshold,
            min_speech_duration_ms=self._min_speech_ms,
            min_silence_duration_ms=self._min_silence_ms,
            return_seconds=True,
        )
        segments: List[VADSegment] = [
            VADSegment(float(s["start"]), float(s["end"])) for s in stamps
        ]
        total = len(audio) / sample_rate if sample_rate > 0 else 0.0
        speech = sum(seg.duration for seg in segments)
        ratio = speech / total if total > 0 else 0.0
        return VADResult(segments=segments, speech_ratio=ratio, backend=self.name)

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._get_speech_timestamps = None
            self._loaded = False
