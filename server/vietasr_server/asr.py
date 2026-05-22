"""Pipeline engines for the HTTP/WS handlers.

A request may pick any preset or any custom set of modules, so pipelines are
built on demand and cached by an EngineRegistry. The underlying C++ core clones
per-Session state, so one cached engine safely serves concurrent batch
transcriptions and concurrent streaming Sessions.
"""
from __future__ import annotations

import threading
from typing import List, Optional, Tuple

import numpy as np

from vietasr import Pipeline, Result, Session

MODEL_ID = "vietasr"

# Canonical pipeline order for custom module sets — modules run in this order
# regardless of the order the caller listed them.
MODULE_ORDER = [
    "vad", "diarization", "vietasr", "itn", "punctuation",
    "gender", "emotion", "dialect", "noise",
]


def order_modules(modules: List[str]) -> List[str]:
    """Sort a custom module set into the canonical pipeline order, de-duped."""
    rank = {name: i for i, name in enumerate(MODULE_ORDER)}
    return sorted(dict.fromkeys(modules), key=lambda m: rank.get(m, len(rank)))


class AsrEngine:
    """One built pipeline (a preset or a custom module set)."""

    def __init__(self, *, preset: Optional[str] = None,
                 modules: Optional[List[str]] = None):
        if modules:
            ordered = order_modules(modules)
            pipeline = Pipeline.new()
            for name in ordered:
                pipeline.add(name)
            pipeline.build()
            self.spec: Tuple[str, ...] = ("modules", *ordered)
            self.description = "modules:" + "+".join(ordered)
        else:
            preset = preset or "transcribe"
            pipeline = Pipeline.preset(preset)
            self.spec = ("preset", preset)
            self.description = f"preset:{preset}"
        self._pipeline = pipeline

    def transcribe(self, pcm: np.ndarray, sample_rate: float) -> Result:
        """Batch transcription of a complete clip (non-streaming)."""
        return self._pipeline.transcribe(pcm, sample_rate)

    def new_session(self, sample_rate: float) -> Session:
        """Open a streaming session; caller owns its lifetime."""
        return self._pipeline.stream(sample_rate)

    def active_execution_provider(self) -> str:
        """ONNX Runtime execution provider the engine runs on — reflects the
        auto-detect + fallback outcome (e.g. "CPUExecutionProvider")."""
        session = self._pipeline.stream(16000.0)
        try:
            return session.active_execution_provider()
        finally:
            session.close()

    def close(self) -> None:
        self._pipeline.close()


class EngineRegistry:
    """Builds pipelines on demand and caches them, keyed by preset name or
    custom module set. The cache is unbounded — a test console only explores a
    small, fixed set of pipelines, and evicting an engine while a streaming
    Session still references it would be unsafe.
    """

    def __init__(self, *, default_preset: str = "transcribe",
                 default_modules: Optional[List[str]] = None):
        self._lock = threading.Lock()
        self._cache: dict = {}
        if default_modules:
            self._default_key: Tuple[str, ...] = (
                "modules", *order_modules(default_modules))
        else:
            self._default_key = ("preset", default_preset or "transcribe")
        # Build the default eagerly so startup can report the runtime.
        self.default = self.resolve(
            preset=default_preset, modules=default_modules)

    def resolve(self, *, preset: Optional[str] = None,
                modules: Optional[List[str]] = None) -> AsrEngine:
        """Return the engine for a preset or a custom module set, building and
        caching it on first use. With neither argument, returns the default."""
        if modules:
            key: Tuple[str, ...] = ("modules", *order_modules(modules))
        elif preset:
            key = ("preset", preset)
        else:
            key = self._default_key

        with self._lock:
            engine = self._cache.get(key)
        if engine is not None:
            return engine

        # Build outside the lock — building loads the model and can be slow.
        engine = AsrEngine(
            preset=(key[1] if key[0] == "preset" else None),
            modules=(list(key[1:]) if key[0] == "modules" else None))

        with self._lock:
            existing = self._cache.get(key)
            if existing is not None:
                engine.close()  # lost a build race — keep the cached engine
                return existing
            self._cache[key] = engine
        return engine

    def close(self) -> None:
        with self._lock:
            for engine in self._cache.values():
                engine.close()
            self._cache.clear()


def available_pipelines() -> dict:
    """Presets and modules the server can build, for the test console UI."""
    modules = Pipeline.list_modules()
    return {
        "presets": Pipeline.list_presets(),
        "modules": order_modules(modules) if modules else [],
    }
