from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .base import (
    BaseVADBackend,
    VADBackendError,
    VADResult,
    dynamic_merge_segments,
    normalize_silence_schedule,
)


class DynamicVADBackend(BaseVADBackend):
    """FunASR-style dynamic VAD as a backend-agnostic wrapper.

    Wraps any registered VAD provider, runs its detection, then re-cuts the output
    segments with a duration-adaptive silence threshold: the silence gap needed to
    cut shrinks as speech accumulates since the last cut (short utterances tolerate
    longer silences; long runs cut on even short silences). The schedule is a list of
    ``(accumulated_ms_limit, silence_threshold_ms)`` tuples.

    Output is an ordinary ``VADResult`` so this plugs into VADStage and
    VADDiarizeStage exactly like a single backend, including over ``consensus``.
    """

    name = "dynamic"

    DEFAULT_SILENCE_SCHEDULE = [
        (5000, 2000),
        (10000, 1500),
        (15000, 1000),
        (30000, 800),
        (45000, 400),
        (1e9, 100),
    ]

    def __init__(
        self,
        provider: str = "silero",
        provider_kwargs: Optional[Dict] = None,
        silence_schedule: Optional[List[Tuple[float, float]]] = None,
        min_dur_s: float = 0.0,
        pad_s: float = 0.0,
        hf_token: Optional[str] = None,
    ):
        self._provider_name = provider
        self._provider_kwargs = dict(provider_kwargs or {})
        self._schedule = normalize_silence_schedule(
            silence_schedule or self.DEFAULT_SILENCE_SCHEDULE
        )
        self._min_dur_s = min_dur_s
        self._pad_s = pad_s
        self._hf_token = hf_token
        self._backend: Optional[BaseVADBackend] = None
        self._loaded = False

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        from . import build_vad_backend

        kwargs = dict(self._provider_kwargs)
        if self._provider_name in ("pyannote_seg", "consensus") and self._hf_token:
            kwargs.setdefault("hf_token", self._hf_token)

        self._backend = build_vad_backend(self._provider_name, **kwargs)
        self._backend.load(device=device)
        self._loaded = True
        logger.info(
            f"Dynamic VAD ready: provider={self._backend.name} "
            f"schedule={self._schedule}"
        )

    def detect(self, audio: np.ndarray, sample_rate: int) -> VADResult:
        if not self._loaded or self._backend is None:
            raise VADBackendError("DynamicVADBackend not loaded. Call load() first.")

        audio = np.asarray(audio, dtype=np.float32)
        base = self._backend.detect(audio, sample_rate)
        segments = dynamic_merge_segments(
            base.segments,
            self._schedule,
            min_dur_s=self._min_dur_s,
            pad_s=self._pad_s,
        )

        total = len(audio) / sample_rate if sample_rate > 0 else 0.0
        speech = sum(seg.duration for seg in segments)
        ratio = speech / total if total > 0 else 0.0
        return VADResult(
            segments=segments,
            speech_ratio=ratio,
            backend=self.name,
            raw={
                "provider": self._backend.name,
                "schedule": self._schedule,
                "base_segments": len(base.segments),
                "dynamic_segments": len(segments),
            },
        )

    def unload(self) -> None:
        if self._backend is not None:
            self._backend.unload()
        self._backend = None
        self._loaded = False
