import numpy as np
import pytest

from multitalker_asr.data.pipeline.checkpoint import PipelineCheckpoint
from multitalker_asr.data.pipeline.config import PipelineConfig
from multitalker_asr.data.pipeline.quality_backends import (
    QUALITY_REGISTRY,
    build_quality_backend,
    list_quality_backends,
)
from multitalker_asr.data.pipeline.quality_backends.snr import SNREstimatorBackend
from multitalker_asr.data.pipeline.stages.audio_quality import AudioQualityStage

_SR = 16000


def _speech(seconds=2.0, sr=_SR, amp=0.3, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(int(seconds * sr)) / sr
    tone = amp * np.sin(2 * np.pi * 180 * t)
    gate = (np.sin(2 * np.pi * 3 * t) > 0).astype(np.float64)
    return (tone * gate + 1e-4 * rng.randn(t.size)).astype(np.float32)


def _noise(seconds=2.0, sr=_SR, amp=0.3, seed=1):
    rng = np.random.RandomState(seed)
    return (amp * rng.randn(int(seconds * sr))).astype(np.float32)


class TestSNRBackend:
    def test_registry(self):
        assert "snr" in QUALITY_REGISTRY
        assert "snr" in list_quality_backends()
        assert isinstance(build_quality_backend("snr"), SNREstimatorBackend)

    def test_clean_speech_high_snr(self):
        be = SNREstimatorBackend()
        res = be.score(_speech(), _SR)
        assert res.snr_db > 15.0
        assert 0.0 <= res.score <= 1.0
        assert res.backend == "snr"

    def test_pure_noise_low_snr(self):
        be = SNREstimatorBackend()
        res = be.score(_noise(), _SR)
        assert res.snr_db < 6.0

    def test_clean_beats_noise(self):
        be = SNREstimatorBackend()
        assert be.score(_speech(), _SR).snr_db > be.score(_noise(), _SR).snr_db

    def test_empty_audio_safe(self):
        res = SNREstimatorBackend().score(np.array([], dtype=np.float32), _SR)
        assert res.snr_db == 0.0 and res.score == 0.0

    def test_unknown_backend_raises(self):
        with pytest.raises(Exception):
            build_quality_backend("nope")


class _StubLoader:
    def __init__(self, mapping):
        self._mapping = mapping

    def load(self, path):
        return self._mapping[path], _SR


class TestAudioQualityStage:
    def _stage(self, mapping):
        stage = AudioQualityStage(sample_rate=_SR)
        stage._audio_loader = _StubLoader(mapping)
        return stage

    def _run(self, stage, records, cfg, tmp_path):
        ck = PipelineCheckpoint(checkpoint_dir=str(tmp_path / "ck"))
        return stage.run(records, cfg, ck)

    def test_drops_noisy_keeps_clean(self, tmp_path):
        mapping = {"clean.wav": _speech(), "noisy.wav": _noise()}
        recs = [
            {"id": "1", "audio_filepath": "clean.wav", "extra": {}},
            {"id": "2", "audio_filepath": "noisy.wav", "extra": {}},
        ]
        cfg = PipelineConfig(output_dir=str(tmp_path))
        cfg.audio_quality.min_snr_db = 6.0
        out = self._run(self._stage(mapping), recs, cfg, tmp_path)
        ids = {r["id"] for r in out}
        assert "1" in ids and "2" not in ids
        kept = next(r for r in out if r["id"] == "1")
        assert "snr_db" in kept["extra"] and "quality_score" in kept["extra"]

    def test_disabled_passthrough(self, tmp_path):
        recs = [{"id": "1", "audio_filepath": "x.wav", "extra": {}}]
        cfg = PipelineConfig(output_dir=str(tmp_path))
        cfg.audio_quality.enabled = False
        out = self._run(self._stage({"x.wav": _noise()}), recs, cfg, tmp_path)
        assert out == recs

    def test_load_failure_keeps_record(self, tmp_path):
        recs = [{"id": "1", "audio_filepath": "missing.wav", "extra": {}}]
        cfg = PipelineConfig(output_dir=str(tmp_path))
        out = self._run(self._stage({}), recs, cfg, tmp_path)
        assert len(out) == 1 and out[0]["id"] == "1"
