"""
Tests for AlignStage — backend-driven forced alignment, transcript source
selection, alignment-driven trim, checkpoint skip, and GPU cleanup.
"""

from pathlib import Path
from unittest.mock import patch

from multitalker_asr.data.pipeline.align_backends import AlignedWord, AlignResult
from multitalker_asr.data.pipeline.checkpoint import PipelineCheckpoint
from multitalker_asr.data.pipeline.config import PipelineConfig
from multitalker_asr.data.pipeline.stages.align import AlignStage


class _FakeAlignBackend:
    def __init__(self, result=None):
        self._result = result or AlignResult(words=[], score=0.0)
        self.calls = []
        self.unloaded = False

    def align(self, audio_path, text, language):
        self.calls.append((audio_path, text, language))
        return self._result

    def unload(self):
        self.unloaded = True


def _make_checkpoint(tmp_path: Path) -> PipelineCheckpoint:
    return PipelineCheckpoint(str(tmp_path / "checkpoints"))


def _make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(output_dir=str(tmp_path))


def _run_with_backend(stage, backend, records, cfg, cp):
    with patch.object(stage, "_ensure_loaded"):
        stage._backend = backend
        return stage.run(records, cfg, cp)


def test_align_empty_text_skipped(tmp_path):
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)
    records = [{"id": "r1", "audio_filepath": "/fake.wav", "text": "", "text_itn": ""}]
    stage = AlignStage()
    backend = _FakeAlignBackend()

    result = _run_with_backend(stage, backend, records, cfg, cp)

    assert backend.calls == []
    assert result[0]["alignment"] == []
    assert result[0]["alignment_score"] == 0.0


def test_align_checkpoint_skip(tmp_path):
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)
    cp.mark_processed("r1", "align")
    records = [{"id": "r1", "audio_filepath": "/fake.wav", "text_itn": "hello"}]
    stage = AlignStage()
    backend = _FakeAlignBackend()

    _run_with_backend(stage, backend, records, cfg, cp)

    assert backend.calls == []


def test_align_uses_text_itn_by_default(tmp_path):
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)
    records = [
        {"id": "r1", "audio_filepath": "/fake.wav", "text": "spoken", "text_itn": "ensembled"}
    ]
    stage = AlignStage()
    backend = _FakeAlignBackend()

    _run_with_backend(stage, backend, records, cfg, cp)

    assert backend.calls == [("/fake.wav", "ensembled", cfg.align.language)]


def test_align_fallback_to_text(tmp_path):
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)
    records = [{"id": "r1", "audio_filepath": "/fake.wav", "text": "xin chào", "text_itn": None}]
    stage = AlignStage()
    backend = _FakeAlignBackend()

    _run_with_backend(stage, backend, records, cfg, cp)

    assert backend.calls == [("/fake.wav", "xin chào", cfg.align.language)]


def test_align_writes_alignment_records(tmp_path):
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)
    result = AlignResult(
        words=[AlignedWord("xin", 0.0, 0.3), AlignedWord("chào", 0.3, 0.8)], score=1.0
    )
    records = [{"id": "r1", "audio_filepath": "/fake.wav", "text_itn": "xin chào"}]
    stage = AlignStage()

    out = _run_with_backend(stage, _FakeAlignBackend(result), records, cfg, cp)

    assert out[0]["alignment_score"] == 1.0
    assert out[0]["alignment"][0] == {"text": "xin", "start_time": 0.0, "end_time": 0.3}


def test_align_trim_to_words(tmp_path):
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)
    cfg.align.trim_to_words = True
    cfg.align.trim_pad_s = 0.0
    result = AlignResult(
        words=[AlignedWord("a", 0.5, 0.9), AlignedWord("b", 0.9, 1.5)], score=1.0
    )
    records = [
        {"id": "r1", "audio_filepath": "/fake.wav", "text_itn": "a b", "duration": 2.0, "offset": 0.0}
    ]
    stage = AlignStage()

    out = _run_with_backend(stage, _FakeAlignBackend(result), records, cfg, cp)

    assert out[0]["offset"] == 0.5
    assert out[0]["duration"] == 1.0
    assert out[0]["extra"]["align_trim"]["start"] == 0.5


def test_align_gpu_freed(tmp_path):
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)
    records = [{"id": "r1", "audio_filepath": "/fake.wav", "text_itn": "test"}]
    stage = AlignStage()
    backend = _FakeAlignBackend()

    _run_with_backend(stage, backend, records, cfg, cp)

    assert stage._backend is None
    assert backend.unloaded is True
