from typing import List, Optional

import numpy as np
from loguru import logger

from .base import ASRBackendError, ASRResult, BaseASRBackend, WordTiming


class VietASRBackend(BaseASRBackend):
    """On-device Vietnamese ASR via the `viet-asr` SDK (embedded ONNX model).

    The SDK ships the model inside the wheel (`pip install viet-asr`) and runs
    fully offline on CPU. Output is plain lowercase text (no punctuation); use it
    as an ensemble vote / cross-check provider, or pair with a PnC/ITN stage.
    """

    name = "vietasr"
    languages = ["vi"]

    def __init__(
        self,
        preset: str = "transcribe",
        model_dir: Optional[str] = None,
        sdk_backend: Optional[str] = None,
    ):
        self._preset = preset
        self._model_dir = model_dir
        self._sdk_backend = sdk_backend
        self._pipeline = None
        self._loaded = False

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        try:
            import vietasr
        except ImportError as exc:
            raise ASRBackendError(
                "viet-asr is required for VietASRBackend. Install via `uv add viet-asr`."
            ) from exc

        logger.info(f"Loading VietASR SDK (preset={self._preset})")
        pipeline = vietasr.Pipeline.preset(self._preset)
        if self._sdk_backend:
            pipeline.set_backend(self._sdk_backend)
        if self._model_dir:
            pipeline.set_model_dir(self._model_dir).build()
        self._pipeline = pipeline
        self._loaded = True

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> ASRResult:
        if not self._loaded:
            raise ASRBackendError("VietASRBackend not loaded. Call load() first.")

        pcm = self._to_int16(audio)
        result = self._pipeline.transcribe(pcm, sample_rate=float(sample_rate))
        return ASRResult(
            text=(result.text or "").strip(),
            confidence=1.0,
            language="vi",
            word_timings=self._parse_segments(result.segments),
            backend=self.name,
            raw={"segments": result.segments} if result.segments else {},
        )

    def _to_int16(self, audio: np.ndarray) -> np.ndarray:
        arr = np.asarray(audio)
        if np.issubdtype(arr.dtype, np.integer):
            return arr.astype(np.int16)
        return (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)

    def _parse_segments(self, segments) -> Optional[List[WordTiming]]:
        timings: List[WordTiming] = []
        for seg in segments or []:
            word = seg.get("text") or seg.get("word")
            if word is None:
                continue
            timings.append(
                WordTiming(
                    word=str(word),
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    confidence=seg.get("confidence"),
                )
            )
        return timings or None

    def unload(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.close()
            except Exception:
                pass
            self._pipeline = None
            self._loaded = False
            logger.info("VietASRBackend unloaded")
