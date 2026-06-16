import math
from typing import Dict, List, Optional, Union

import numpy as np
from loguru import logger

from .base import (
    BaseVADBackend,
    VADBackendError,
    VADResult,
    VADSegment,
    assemble_frames,
    merge_segments,
)


class ConsensusVADBackend(BaseVADBackend):
    """Frame-level consensus over several independent VAD providers.

    Runs each sub-backend, rasterizes their speech segments onto a common frame
    grid, and keeps a frame as speech only when enough providers agree. The
    output is an ordinary ``VADResult`` so this plugs into VADStage and
    VADDiarizeStage exactly like a single backend.

    Strategies map to a vote threshold over the N loaded providers:
      - "majority"     : >= ceil(N / 2)
      - "intersection" : == N   (all agree)
      - "union"        : >= 1    (any)
    An explicit ``min_votes`` overrides the strategy.
    """

    name = "consensus"

    DEFAULT_PROVIDERS = ["silero", "pyannote_seg", "ten"]

    def __init__(
        self,
        providers: Optional[List[Union[str, Dict]]] = None,
        strategy: str = "majority",
        min_votes: Optional[int] = None,
        frame_hop_s: float = 0.02,
        min_gap_s: float = 0.2,
        min_dur_s: float = 0.0,
        pad_s: float = 0.0,
        hf_token: Optional[str] = None,
    ):
        self._provider_specs = providers or list(self.DEFAULT_PROVIDERS)
        self._strategy = strategy.lower()
        self._min_votes = min_votes
        self._frame_hop_s = frame_hop_s
        self._min_gap_s = min_gap_s
        self._min_dur_s = min_dur_s
        self._pad_s = pad_s
        self._hf_token = hf_token
        self._backends: List[BaseVADBackend] = []
        self._loaded = False

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        from . import build_vad_backend

        self._backends = []
        for spec in self._provider_specs:
            backend_name, kwargs = self._parse_spec(spec)
            if backend_name == "pyannote_seg" and self._hf_token:
                kwargs.setdefault("hf_token", self._hf_token)
            try:
                backend = build_vad_backend(backend_name, **kwargs)
                backend.load(device=device)
                self._backends.append(backend)
            except Exception as exc:
                logger.warning(f"Consensus VAD: skipping '{backend_name}': {exc}")

        if not self._backends:
            raise VADBackendError(
                "ConsensusVADBackend loaded no providers. Check installs/config."
            )
        logger.info(
            f"Consensus VAD ready: {[b.name for b in self._backends]} "
            f"strategy={self._strategy} threshold={self._threshold(len(self._backends))}"
        )
        self._loaded = True

    def detect(self, audio: np.ndarray, sample_rate: int) -> VADResult:
        if not self._loaded:
            raise VADBackendError("ConsensusVADBackend not loaded. Call load() first.")

        audio = np.asarray(audio, dtype=np.float32)
        total = len(audio) / sample_rate if sample_rate > 0 else 0.0
        hop_samples = max(1, int(self._frame_hop_s * sample_rate))
        n_frames = max(1, math.ceil(len(audio) / hop_samples)) if len(audio) else 0

        votes = np.zeros(n_frames, dtype=np.int32)
        per_backend_ratio: Dict[str, float] = {}
        for backend in self._backends:
            try:
                result = backend.detect(audio, sample_rate)
            except Exception as exc:
                logger.warning(f"Consensus VAD: '{backend.name}' detect failed: {exc}")
                continue
            per_backend_ratio[backend.name] = round(result.speech_ratio, 4)
            votes += self._rasterize(result.segments, hop_samples, sample_rate, n_frames)

        threshold = self._threshold(len(self._backends))
        flags = (votes >= threshold).tolist()
        segments = assemble_frames(
            flags,
            hop_size=hop_samples,
            sample_rate=sample_rate,
            min_gap_s=self._min_gap_s,
            min_dur_s=self._min_dur_s,
        )
        if self._pad_s > 0.0:
            segments = merge_segments(
                segments, min_gap_s=self._min_gap_s, min_dur_s=self._min_dur_s, pad_s=self._pad_s
            )

        speech = sum(seg.duration for seg in segments)
        ratio = speech / total if total > 0 else 0.0
        return VADResult(
            segments=segments,
            speech_ratio=ratio,
            backend=self.name,
            raw={
                "strategy": self._strategy,
                "threshold": threshold,
                "n_providers": len(self._backends),
                "per_backend_ratio": per_backend_ratio,
            },
        )

    def unload(self) -> None:
        for backend in self._backends:
            backend.unload()
        self._backends = []
        self._loaded = False

    # ---- helpers ----------------------------------------------------------

    def _parse_spec(self, spec: Union[str, Dict]):
        if isinstance(spec, str):
            return spec, {}
        name = spec.get("name")
        if not name:
            raise VADBackendError(f"Provider spec missing 'name': {spec}")
        return name, dict(spec.get("kwargs") or {})

    def _threshold(self, n: int) -> int:
        if self._min_votes is not None:
            return max(1, min(n, self._min_votes))
        if self._strategy == "intersection":
            return n
        if self._strategy == "union":
            return 1
        if self._strategy == "majority":
            return math.ceil(n / 2)
        raise VADBackendError(
            f"Unknown consensus strategy '{self._strategy}'. "
            f"Valid: majority, intersection, union (or set min_votes)."
        )

    def _rasterize(
        self,
        segments: List[VADSegment],
        hop_samples: int,
        sample_rate: int,
        n_frames: int,
    ) -> np.ndarray:
        cover = np.zeros(n_frames, dtype=np.int32)
        frame_dur = hop_samples / sample_rate if sample_rate > 0 else 0.0
        if frame_dur <= 0.0 or n_frames == 0:
            return cover
        for seg in segments:
            lo = max(0, int(seg.start / frame_dur))
            hi = min(n_frames, int(math.ceil(seg.end / frame_dur)))
            if hi > lo:
                cover[lo:hi] = 1
        return cover
