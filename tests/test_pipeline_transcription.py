"""
Tests for Phase 3: TranscribeStage — tag parsing, checkpoint skip, GPU cleanup.
"""

import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from multitalker_asr.data.pipeline.checkpoint import PipelineCheckpoint
from multitalker_asr.data.pipeline.config import PipelineConfig
from multitalker_asr.data.pipeline.stages.transcribe import (
    TranscribeStage,
    _parse_emotion_tag,
    _parse_language_tag,
)


def _make_checkpoint(tmp_path: Path) -> PipelineCheckpoint:
    return PipelineCheckpoint(str(tmp_path / "checkpoints"))


def _make_config(tmp_path: Path, itn_only: bool = False) -> PipelineConfig:
    cfg = PipelineConfig(output_dir=str(tmp_path))
    cfg.transcribe.itn_only = itn_only
    return cfg


# ---------------------------------------------------------------------------
# Tag parsing (pure function tests — no model needed)
# ---------------------------------------------------------------------------

def test_transcribe_emotion_tag_parsing():
    """Emotion tag strings parse to lowercase labels; missing → 'neutral'."""
    assert _parse_emotion_tag("<|HAPPY|>") == "happy"
    assert _parse_emotion_tag("<|SAD|>") == "sad"
    assert _parse_emotion_tag("<|ANGRY|>") == "angry"
    assert _parse_emotion_tag("<|NEUTRAL|>") == "neutral"
    assert _parse_emotion_tag(None) == "neutral"
    assert _parse_emotion_tag("") == "neutral"
    assert _parse_emotion_tag("no_tag_here") == "neutral"


def test_transcribe_language_tag_parsing():
    """Language tag strings parse to ISO codes; missing → 'vi'."""
    assert _parse_language_tag("<|vi|>") == "vi"
    assert _parse_language_tag("<|en|>") == "en"
    assert _parse_language_tag("<|zh|>") == "zh"
    assert _parse_language_tag(None) == "vi"
    assert _parse_language_tag("") == "vi"


# ---------------------------------------------------------------------------
# itn_only mode preserves existing emotion
# ---------------------------------------------------------------------------

def test_transcribe_itn_only_preserves_emotion(tmp_path):
    """In itn_only mode, existing 'emotion' field is kept unchanged."""
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path, itn_only=True)

    records = [{"id": "r1", "audio_filepath": "/fake.wav", "text": "xin chào", "emotion": "happy"}]

    stage = TranscribeStage()
    mock_model = MagicMock()
    mock_model.generate.return_value = [{"text": "xin chào ."}]

    with patch.object(stage, "_load_model"):
        stage._model = mock_model
        result = stage.run(records, cfg, cp)

    assert result[0]["emotion"] == "happy"
    assert result[0]["text_itn"] is not None


# ---------------------------------------------------------------------------
# Checkpoint skip
# ---------------------------------------------------------------------------

def test_transcribe_checkpoint_skip(tmp_path):
    """Already-processed record skips _transcribe_audio entirely."""
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)
    cp.mark_processed("r1", "transcribe")

    records = [{"id": "r1", "audio_filepath": "/fake.wav"}]
    stage = TranscribeStage()

    with patch.object(stage, "_transcribe_audio") as mock_transcribe, \
         patch.object(stage, "_load_model"):
        stage._model = MagicMock()
        result = stage.run(records, cfg, cp)
        mock_transcribe.assert_not_called()


# ---------------------------------------------------------------------------
# Missing audio skipped
# ---------------------------------------------------------------------------

def test_transcribe_missing_audio_skipped(tmp_path):
    """Record with non-existent audio_filepath sets text='' and continues."""
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)

    records = [{"id": "r_miss", "audio_filepath": "/nonexistent/audio.wav"}]
    stage = TranscribeStage()

    with patch.object(stage, "_load_model"):
        stage._model = MagicMock()
        result = stage.run(records, cfg, cp)

    assert len(result) == 1
    assert result[0]["text"] == ""


# ---------------------------------------------------------------------------
# GPU freed
# ---------------------------------------------------------------------------

def test_transcribe_gpu_freed(tmp_path):
    """After run(), self._model is None (GPU freed)."""
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path, itn_only=True)

    records = [{"id": "r1", "audio_filepath": "/fake.wav", "text": "hello"}]
    stage = TranscribeStage()

    mock_model = MagicMock()
    mock_model.generate.return_value = [{"text": "hello ."}]

    with patch.object(stage, "_load_model"):
        stage._model = mock_model
        stage.run(records, cfg, cp)

    assert stage._model is None
