"""Thin wrapper around the VietASR pipeline used by the HTTP/WS handlers.

One AsrEngine (one built Pipeline) is shared by the whole process. The
underlying C++ core clones per-Session state, so concurrent batch
transcriptions and concurrent streaming Sessions are safe.
"""
from __future__ import annotations

import numpy as np

from vietasr import Pipeline, Result, Session

MODEL_ID = "vietasr"


class AsrEngine:
    def __init__(self, preset: str = "transcribe", modules: list[str] | None = None):
        if modules:
            pipeline = Pipeline.new()
            for name in modules:
                pipeline.add(name)
            pipeline.build()
            self.description = "+".join(modules)
        else:
            pipeline = Pipeline.preset(preset)
            self.description = f"preset:{preset}"
        self._pipeline = pipeline

    def transcribe(self, pcm: np.ndarray, sample_rate: float) -> Result:
        """Batch transcription of a complete clip (non-streaming)."""
        return self._pipeline.transcribe(pcm, sample_rate)

    def new_session(self, sample_rate: float) -> Session:
        """Open a streaming session; caller owns its lifetime."""
        return self._pipeline.stream(sample_rate)

    def active_execution_provider(self) -> str:
        """ONNX Runtime execution provider the engine runs on — e.g.
        "CUDAExecutionProvider" on the GPU image, "CPUExecutionProvider" on the
        CPU/edge image. Reflects the auto-detect + fallback outcome."""
        session = self._pipeline.stream(16000.0)
        try:
            return session.active_execution_provider()
        finally:
            session.close()

    def close(self) -> None:
        self._pipeline.close()
