import numpy as np
import pytest

from multitalker_asr.data.pipeline.checkpoint import PipelineCheckpoint
from multitalker_asr.data.pipeline.config import PipelineConfig
from multitalker_asr.data.pipeline.speaker import (
    centroids,
    cluster_embeddings,
    cluster_margin,
    estimate_threshold,
)
from multitalker_asr.data.pipeline.stages.speaker_verify import SpeakerVerifyStage


def _voice(seed, dim=32):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


_VOICES = {"A": _voice(1), "B": _voice(2)}


class _FakeEmbedder:
    def __init__(self, *_a, healthy=True, **_kw):
        self._healthy = healthy

    def health(self):
        return self._healthy

    def embed(self, audio_path):
        return _VOICES.get(audio_path.split("/")[-1][0])


@pytest.fixture
def patch_embedder(monkeypatch):
    monkeypatch.setattr(
        "multitalker_asr.data.pipeline.stages.speaker_verify.SpeakerEmbedder",
        lambda url, timeout: _FakeEmbedder(),
    )


class TestClusterMargin:
    def test_well_separated_positive_margin(self):
        a, b = _voice(1), _voice(2)
        embs = np.stack([a, a, b])
        labels = np.array([0, 0, 1])
        cents = centroids(embs, labels)
        assert cluster_margin(a, 0, cents) > 0.0

    def test_single_cluster_returns_own_sim(self):
        a = _voice(1)
        cents = centroids(a[None, :], np.array([0]))
        assert cluster_margin(a, 0, cents) == pytest.approx(1.0, abs=1e-3)


class TestEstimateThreshold:
    def test_two_clear_clusters_in_band(self):
        a, b = _voice(1), _voice(2)
        embs = np.stack([a + 0.01 * _voice(10), a + 0.01 * _voice(11), a,
                         b + 0.01 * _voice(12), b])
        thr = estimate_threshold(embs)
        assert 0.30 <= thr <= 0.65

    def test_too_few_returns_default(self):
        assert estimate_threshold(_voice(1)[None, :], default=0.42) == 0.42


class TestSpeakerVerifyConfidence:
    def _seg(self, sid, path, source="src1", overlap=False):
        return {
            "id": sid, "audio_filepath": path, "speaker_id": "SPEAKER_00",
            "source_audio": source, "duration": 3.0,
            "extra": {"is_overlap": overlap},
        }

    def _run(self, recs, cfg, tmp_path):
        ck = PipelineCheckpoint(checkpoint_dir=str(tmp_path / "ck"))
        return SpeakerVerifyStage().run(recs, cfg, ck)

    def test_stores_confidence_fields(self, patch_embedder, tmp_path):
        cfg = PipelineConfig(output_dir=str(tmp_path))
        cfg.speaker_verify.cluster_threshold = 0.6
        recs = [self._seg("1", "A0.wav"), self._seg("2", "A1.wav"),
                self._seg("3", "B0.wav")]
        out = self._run(recs, cfg, tmp_path)
        for r in out:
            e = r["extra"]
            assert "cluster_centroid_sim" in e
            assert "cluster_margin" in e
            assert "cluster_threshold" in e
            assert "cluster_size" in e
            assert e["cluster_threshold"] == pytest.approx(0.6)

    def test_auto_threshold_per_file(self, patch_embedder, tmp_path):
        cfg = PipelineConfig(output_dir=str(tmp_path))
        cfg.speaker_verify.cluster_threshold = "auto"
        recs = [self._seg("1", "A0.wav"), self._seg("2", "A1.wav"),
                self._seg("3", "B0.wav"), self._seg("4", "B1.wav")]
        out = self._run(recs, cfg, tmp_path)
        thr = {r["extra"]["cluster_threshold"] for r in out}
        assert len(thr) == 1
        used = thr.pop()
        assert 0.30 <= used <= 0.65
        assert all(r["num_speakers"] == 2 for r in out)
