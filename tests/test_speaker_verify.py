import numpy as np
import pytest

from multitalker_asr.data.pipeline import stages
from multitalker_asr.data.pipeline.checkpoint import PipelineCheckpoint
from multitalker_asr.data.pipeline.config import PipelineConfig
from multitalker_asr.data.pipeline.stages.speaker_verify import SpeakerVerifyStage


def _voice(seed, dim=32):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


_VOICES = {"A": _voice(1), "B": _voice(2)}


class _FakeEmbedder:
    """Maps audio_filepath prefix (A/B) to a fixed voice; None for 'FAIL'."""

    def __init__(self, *_args, healthy=True, **_kw):
        self._healthy = healthy

    def health(self):
        return self._healthy

    def embed(self, audio_path):
        key = audio_path.split("/")[-1][0]
        if key == "F":
            return None
        return _VOICES.get(key)


@pytest.fixture
def patch_embedder(monkeypatch):
    def _apply(healthy=True):
        monkeypatch.setattr(
            "multitalker_asr.data.pipeline.stages.speaker_verify.SpeakerEmbedder",
            lambda url, timeout: _FakeEmbedder(healthy=healthy),
        )
    return _apply


def _seg(sid, path, speaker, source="src1", overlap=False):
    return {
        "id": sid, "audio_filepath": path, "speaker_id": speaker,
        "source_audio": source, "duration": 3.0,
        "extra": {"is_overlap": overlap},
    }


def _cfg(tmp_path):
    c = PipelineConfig(output_dir=str(tmp_path))
    c.speaker_verify.cluster_threshold = 0.6
    return c


def _run(records, cfg, tmp_path):
    ck = PipelineCheckpoint(checkpoint_dir=str(tmp_path / "ck"))
    return SpeakerVerifyStage().run(records, cfg, ck)


class TestSpeakerVerify:
    def test_relabel_two_voices(self, patch_embedder, tmp_path):
        patch_embedder()
        recs = [
            _seg("1", "A0.wav", "SPEAKER_00"),
            _seg("2", "A1.wav", "SPEAKER_00"),
            _seg("3", "A2.wav", "SPEAKER_00"),
            _seg("4", "B0.wav", "SPEAKER_00"),   # actually voice B, mislabeled
        ]
        out = _run(recs, _cfg(tmp_path), tmp_path)
        assert len(out) == 4
        assert all(r["num_speakers"] == 2 for r in out)
        spk = {r["audio_filepath"]: r["speaker_id"] for r in out}
        assert spk["A0.wav"] == spk["A1.wav"] == spk["A2.wav"]
        assert spk["B0.wav"] != spk["A0.wav"]          # B separated/relabeled
        assert all(r["speaker_id"].startswith("SPK_") for r in out)

    def test_service_down_passthrough(self, patch_embedder, tmp_path):
        patch_embedder(healthy=False)
        recs = [_seg("1", "A0.wav", "SPEAKER_00")]
        out = _run(recs, _cfg(tmp_path), tmp_path)
        assert out == recs  # unchanged

    def test_embed_failure_passthrough(self, patch_embedder, tmp_path):
        patch_embedder()
        recs = [
            _seg("1", "A0.wav", "SPEAKER_00"),
            _seg("2", "A1.wav", "SPEAKER_00"),
            _seg("3", "FAIL.wav", "SPEAKER_00"),  # embed returns None
        ]
        out = _run(recs, _cfg(tmp_path), tmp_path)
        assert len(out) == 3  # failed one kept (not dropped)
        consistency = [r.get("extra", {}).get("speaker_consistency") for r in out]
        assert any(c is not None for c in consistency)

    def test_overlap_segment_kept(self, patch_embedder, tmp_path):
        patch_embedder()
        cfg = _cfg(tmp_path)
        cfg.speaker_verify.ambiguous_max = 1.0   # would drop everything non-overlap
        cfg.speaker_verify.relabel_min = 1.0
        recs = [
            _seg("1", "A0.wav", "SPEAKER_00"),
            _seg("2", "A1.wav", "SPEAKER_00"),
            _seg("3", "B0.wav", "SPEAKER_00", overlap=True),  # overlap → keep
        ]
        out = _run(recs, cfg, tmp_path)
        assert any(r["audio_filepath"] == "B0.wav" for r in out)  # overlap survived
