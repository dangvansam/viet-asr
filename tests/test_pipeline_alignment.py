"""
Tests for Phase 4: AlignStage — score calculation, checkpoint skip, GPU cleanup.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from multitalker_asr.data.pipeline.checkpoint import PipelineCheckpoint
from multitalker_asr.data.pipeline.config import PipelineConfig
from multitalker_asr.data.pipeline.stages.align import AlignStage


def _make_checkpoint(tmp_path: Path) -> PipelineCheckpoint:
    return PipelineCheckpoint(str(tmp_path / "checkpoints"))


def _make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(output_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# _calculate_alignment_score
# ---------------------------------------------------------------------------

def test_align_score_valid():
    """All valid timestamps → score == 1.0."""
    stage = AlignStage()
    alignment = [
        {"text": "xin", "start_time": 0.0, "end_time": 0.3},
        {"text": "chào", "start_time": 0.3, "end_time": 0.7},
        {"text": "bạn", "start_time": 0.7, "end_time": 1.1},
    ]
    assert stage._calculate_alignment_score(alignment) == 1.0


def test_align_score_partial():
    """2 of 4 valid → score == 0.5."""
    stage = AlignStage()
    alignment = [
        {"text": "a", "start_time": 0.0, "end_time": 0.3},   # valid
        {"text": "b", "start_time": 0.3, "end_time": 0.7},   # valid
        {"text": "c", "start_time": -1.0, "end_time": 0.5},  # invalid (start < 0)
        {"text": "d", "start_time": 1.0, "end_time": 0.8},   # invalid (end <= start)
    ]
    assert stage._calculate_alignment_score(alignment) == 0.5


def test_align_score_empty():
    """Empty alignment → score == 0.0."""
    stage = AlignStage()
    assert stage._calculate_alignment_score([]) == 0.0


# ---------------------------------------------------------------------------
# Empty text skips aligner
# ---------------------------------------------------------------------------

def test_align_empty_text_skipped(tmp_path):
    """Record with empty text gets alignment=[], score=0.0 without loading aligner."""
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)

    records = [{"id": "r1", "audio_filepath": "/fake.wav", "text": "", "text_itn": ""}]
    stage = AlignStage()

    with patch.object(stage, "_load_aligner") as mock_load, \
         patch.object(stage, "_align_one") as mock_align:
        result = stage.run(records, cfg, cp)
        mock_align.assert_not_called()

    assert result[0]["alignment"] == []
    assert result[0]["alignment_score"] == 0.0


# ---------------------------------------------------------------------------
# Checkpoint skip
# ---------------------------------------------------------------------------

def test_align_checkpoint_skip(tmp_path):
    """Already-processed record does not call _align_one."""
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)
    cp.mark_processed("r1", "align")

    records = [{"id": "r1", "audio_filepath": "/fake.wav", "text_itn": "hello"}]
    stage = AlignStage()

    with patch.object(stage, "_load_aligner"), \
         patch.object(stage, "_align_one") as mock_align:
        stage._aligner = MagicMock()
        result = stage.run(records, cfg, cp)
        mock_align.assert_not_called()


# ---------------------------------------------------------------------------
# Fallback to 'text' when 'text_itn' is None
# ---------------------------------------------------------------------------

def test_align_fallback_to_text(tmp_path):
    """When text_itn is None/empty, falls back to text field for alignment."""
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)

    records = [{"id": "r1", "audio_filepath": "/fake.wav", "text": "xin chào", "text_itn": None}]
    stage = AlignStage()

    with patch.object(stage, "_load_aligner"), \
         patch.object(stage, "_align_one", return_value=[]) as mock_align:
        stage._aligner = MagicMock()
        stage.run(records, cfg, cp)
        mock_align.assert_called_once_with("/fake.wav", "xin chào", cfg.align.language)


# ---------------------------------------------------------------------------
# GPU freed
# ---------------------------------------------------------------------------

def test_align_gpu_freed(tmp_path):
    """After run(), self._aligner is None (GPU freed)."""
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)

    records = [{"id": "r1", "audio_filepath": "/fake.wav", "text_itn": "test"}]
    stage = AlignStage()

    with patch.object(stage, "_load_aligner"), \
         patch.object(stage, "_align_one", return_value=[]):
        stage._aligner = MagicMock()
        stage.run(records, cfg, cp)

    assert stage._aligner is None
