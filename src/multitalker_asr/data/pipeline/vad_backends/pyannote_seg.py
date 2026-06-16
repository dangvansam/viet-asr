from typing import List, Optional

import numpy as np
from loguru import logger

from .base import BaseVADBackend, VADBackendError, VADResult, VADSegment


class PyannoteSegmentationVADBackend(BaseVADBackend):
    name = "pyannote_seg"

    def __init__(
        self,
        model_id: str = "pyannote/segmentation-3.0",
        hf_token: Optional[str] = None,
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
    ):
        self._model_id = model_id
        self._hf_token = hf_token
        self._min_duration_on = min_duration_on
        self._min_duration_off = min_duration_off
        self._pipeline = None
        self._loaded = False

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        try:
            from pyannote.audio import Model
            from pyannote.audio.pipelines import VoiceActivityDetection
        except ImportError as exc:
            raise VADBackendError(
                "pyannote.audio is required for PyannoteSegmentationVADBackend."
            ) from exc

        logger.info(f"Loading pyannote VAD model '{self._model_id}'")
        # Only pass the auth token when set — offline/cached loads (and pyannote 4.x)
        # break if use_auth_token= is forwarded with None.
        kwargs = {"use_auth_token": self._hf_token} if self._hf_token else {}
        model = Model.from_pretrained(self._model_id, **kwargs)
        pipeline = VoiceActivityDetection(segmentation=model)
        pipeline.instantiate(
            {
                "min_duration_on": self._min_duration_on,
                "min_duration_off": self._min_duration_off,
            }
        )
        import torch

        pipeline.to(torch.device(device))
        self._pipeline = pipeline
        self._loaded = True

    def detect(self, audio: np.ndarray, sample_rate: int) -> VADResult:
        if not self._loaded:
            raise VADBackendError(
                "PyannoteSegmentationVADBackend not loaded. Call load() first."
            )

        import torch

        waveform = torch.as_tensor(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        annotation = self._pipeline(
            {"waveform": waveform, "sample_rate": sample_rate}
        )
        segments: List[VADSegment] = [
            VADSegment(float(seg.start), float(seg.end))
            for seg in annotation.get_timeline()
        ]
        total = len(audio) / sample_rate if sample_rate > 0 else 0.0
        speech = sum(seg.duration for seg in segments)
        ratio = speech / total if total > 0 else 0.0
        return VADResult(segments=segments, speech_ratio=ratio, backend=self.name)

    def unload(self) -> None:
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._loaded = False
