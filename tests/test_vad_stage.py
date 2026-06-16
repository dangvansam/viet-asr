import numpy as np
import pytest
import soundfile as sf

from multitalker_asr.data.pipeline.checkpoint import PipelineCheckpoint
from multitalker_asr.data.pipeline.config import PipelineConfig
from multitalker_asr.data.pipeline.stages.vad import VADStage
from multitalker_asr.data.pipeline.vad_backends import (
    VAD_REGISTRY,
    BaseVADBackend,
    VADResult,
    register_vad_backend,
)


class _EnergyVAD(BaseVADBackend):
    name = "_energy"

    def load(self, device="cpu"):
        self._loaded = True

    def detect(self, audio, sample_rate):
        mask = np.abs(audio) > 0.01
        ratio = float(mask.mean()) if audio.size else 0.0
        spans = [(0.0, len(audio) / sample_rate)] if ratio > 0 else []
        return VADResult.from_spans(spans, len(audio) / sample_rate, backend=self.name)


@pytest.fixture(autouse=True)
def _register_energy_vad():
    register_vad_backend("_energy", _EnergyVAD)
    yield
    VAD_REGISTRY.pop("_energy", None)


def _write_wav(path, audio, sr=16000):
    sf.write(str(path), audio.astype(np.float32), sr)
    return str(path)


def _config(tmp_path):
    cfg = PipelineConfig(output_dir=str(tmp_path))
    cfg.vad.backend = "_energy"
    cfg.vad.min_speech_ratio = 0.15
    cfg.vad.enable_prefilter = True
    return cfg


class TestVADStage:
    def test_drops_silence_keeps_speech(self, tmp_path):
        silence = _write_wav(tmp_path / "silence.wav", np.zeros(16000))
        speech = _write_wav(tmp_path / "speech.wav", np.ones(16000) * 0.5)
        cfg = _config(tmp_path)
        ckpt = PipelineCheckpoint(checkpoint_dir=str(tmp_path / "ckpt"))

        records = [
            {"id": "silence", "audio_filepath": silence},
            {"id": "speech", "audio_filepath": speech},
        ]
        out = VADStage().run(records, cfg, ckpt)
        ids = {r["id"] for r in out}
        assert ids == {"speech"}
        kept = out[0]
        assert kept["speech_ratio"] > 0.15
        assert kept["vad_segments"]

    def test_prefilter_disabled_keeps_all(self, tmp_path):
        silence = _write_wav(tmp_path / "silence.wav", np.zeros(16000))
        cfg = _config(tmp_path)
        cfg.vad.enable_prefilter = False
        ckpt = PipelineCheckpoint(checkpoint_dir=str(tmp_path / "ckpt"))
        out = VADStage().run([{"id": "s", "audio_filepath": silence}], cfg, ckpt)
        assert len(out) == 1

    def test_checkpoint_skip(self, tmp_path):
        speech = _write_wav(tmp_path / "speech.wav", np.ones(16000) * 0.5)
        cfg = _config(tmp_path)
        ckpt = PipelineCheckpoint(checkpoint_dir=str(tmp_path / "ckpt"))
        record = {"id": "speech", "audio_filepath": speech}
        VADStage().run([record], cfg, ckpt)
        assert ckpt.is_processed("speech", "vad")
        again = VADStage().run([record], cfg, ckpt)
        assert len(again) == 1
