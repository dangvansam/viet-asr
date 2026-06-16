from typing import List

import numpy as np
from loguru import logger

from .base import BaseVADBackend, VADBackendError, VADResult, assemble_frames


class TenVADBackend(BaseVADBackend):
    name = "ten"

    def __init__(
        self,
        hop_size: int = 256,
        threshold: float = 0.5,
        min_gap_s: float = 0.2,
        min_dur_s: float = 0.0,
    ):
        self._hop_size = hop_size
        self._threshold = threshold
        self._min_gap_s = min_gap_s
        self._min_dur_s = min_dur_s
        self._vad = None
        self._loaded = False

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        try:
            from ten_vad import TenVad
        except ImportError as exc:
            raise VADBackendError(
                "ten-vad is required for TenVADBackend. Install via "
                "`uv pip install git+https://github.com/TEN-framework/ten-vad.git`."
            ) from exc

        logger.info("Loading ten-vad")
        self._vad = TenVad(hop_size=self._hop_size, threshold=self._threshold)
        self._loaded = True

    def detect(self, audio: np.ndarray, sample_rate: int) -> VADResult:
        if not self._loaded:
            raise VADBackendError("TenVADBackend not loaded. Call load() first.")

        pcm = self._to_int16(audio)
        flags: List[bool] = []
        for start in range(0, len(pcm) - self._hop_size + 1, self._hop_size):
            frame = pcm[start : start + self._hop_size]
            _, flag = self._vad.process(frame)
            flags.append(bool(flag))

        segments = assemble_frames(
            flags,
            hop_size=self._hop_size,
            sample_rate=sample_rate,
            min_gap_s=self._min_gap_s,
            min_dur_s=self._min_dur_s,
        )
        total = len(audio) / sample_rate if sample_rate > 0 else 0.0
        speech = sum(seg.duration for seg in segments)
        ratio = speech / total if total > 0 else 0.0
        return VADResult(segments=segments, speech_ratio=ratio, backend=self.name)

    def _to_int16(self, audio: np.ndarray) -> np.ndarray:
        arr = np.asarray(audio)
        if np.issubdtype(arr.dtype, np.integer):
            return arr.astype(np.int16)
        clipped = np.clip(arr, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16)

    def unload(self) -> None:
        if self._vad is not None:
            del self._vad
            self._vad = None
            self._loaded = False
