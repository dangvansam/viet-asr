"""Unit tests for ConsensusVADBackend frame-level fusion (no models loaded)."""

import numpy as np
import pytest

from multitalker_asr.data.pipeline.vad_backends import build_vad_backend
from multitalker_asr.data.pipeline.vad_backends.base import (
    BaseVADBackend,
    VADResult,
    VADSegment,
)


class FakeVADBackend(BaseVADBackend):
    def __init__(self, name, spans):
        self.name = name
        self._spans = spans
        self._loaded = False

    def load(self, device="cpu"):
        self._loaded = True

    def detect(self, audio, sample_rate):
        total = len(audio) / sample_rate if sample_rate > 0 else 0.0
        return VADResult.from_spans(self._spans, total, backend=self.name)

    def unload(self):
        self._loaded = False


FAKE_SPANS = {
    "a": [(1.0, 3.0)],
    "b": [(1.0, 3.0)],
    "c": [(4.0, 5.0)],
}


@pytest.fixture
def patched_build(monkeypatch):
    def fake_factory(name, **kwargs):
        return FakeVADBackend(name, FAKE_SPANS[name])

    monkeypatch.setattr(
        "multitalker_asr.data.pipeline.vad_backends.build_vad_backend", fake_factory
    )


def _audio(seconds=5.0, sr=1000):
    return np.zeros(int(seconds * sr), dtype=np.float32), sr


def _spans(result):
    return [(round(s.start, 3), round(s.end, 3)) for s in result.segments]


def test_majority_keeps_only_agreed_region(patched_build):
    backend = build_vad_backend(
        "consensus", strategy="majority", providers=["a", "b", "c"], frame_hop_s=0.02
    )
    backend.load()
    audio, sr = _audio()
    result = backend.detect(audio, sr)
    assert _spans(result) == [(1.0, 3.0)]
    assert result.raw["threshold"] == 2
    assert result.raw["n_providers"] == 3


def test_intersection_requires_all(patched_build):
    backend = build_vad_backend(
        "consensus", strategy="intersection", providers=["a", "b", "c"], frame_hop_s=0.02
    )
    backend.load()
    audio, sr = _audio()
    result = backend.detect(audio, sr)
    assert _spans(result) == []


def test_union_keeps_any(patched_build):
    backend = build_vad_backend(
        "consensus", strategy="union", providers=["a", "b", "c"], frame_hop_s=0.02
    )
    backend.load()
    audio, sr = _audio()
    result = backend.detect(audio, sr)
    assert _spans(result) == [(1.0, 3.0), (4.0, 5.0)]


def test_min_votes_overrides_strategy(patched_build):
    backend = build_vad_backend(
        "consensus", strategy="union", min_votes=2, providers=["a", "b", "c"], frame_hop_s=0.02
    )
    backend.load()
    audio, sr = _audio()
    result = backend.detect(audio, sr)
    assert _spans(result) == [(1.0, 3.0)]


def test_skips_failing_provider(patched_build, monkeypatch):
    def fake_factory(name, **kwargs):
        if name == "broken":
            raise RuntimeError("missing dep")
        return FakeVADBackend(name, FAKE_SPANS[name])

    monkeypatch.setattr(
        "multitalker_asr.data.pipeline.vad_backends.build_vad_backend", fake_factory
    )
    backend = build_vad_backend(
        "consensus", strategy="majority", providers=["a", "b", "broken"], frame_hop_s=0.02
    )
    backend.load()
    audio, sr = _audio()
    result = backend.detect(audio, sr)
    assert result.raw["n_providers"] == 2
    assert result.raw["threshold"] == 1
    assert _spans(result) == [(1.0, 3.0)]
