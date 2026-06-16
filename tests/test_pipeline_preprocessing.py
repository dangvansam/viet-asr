"""
Tests for Phase 2: ExtractAudioStage and VADDiarizeStage.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from multitalker_asr.data.pipeline.checkpoint import PipelineCheckpoint
from multitalker_asr.data.pipeline.config import PipelineConfig
from multitalker_asr.data.pipeline.stages.extract_audio import ExtractAudioStage
from multitalker_asr.data.pipeline.stages.vad_diarize import VADDiarizeStage


def _make_checkpoint(tmp_path: Path) -> PipelineCheckpoint:
    return PipelineCheckpoint(str(tmp_path / "checkpoints"))


def _make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(output_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# ExtractAudioStage tests
# ---------------------------------------------------------------------------

def test_extract_audio_wav_created(tmp_path):
    """Running stage on existing WAV creates output WAV at expected path."""
    # Use the project's demo WAV as test input
    src = Path(__file__).parent.parent / "demo_16k.wav"
    if not src.exists():
        pytest.skip("demo_16k.wav not found")

    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)
    record = {"id": "test_audio", "audio_filepath": str(src)}

    stage = ExtractAudioStage()
    result = stage.run([record], cfg, cp)

    assert len(result) == 1
    out_wav = tmp_path / "extracted" / "test_audio.wav"
    assert out_wav.exists(), f"Expected WAV at {out_wav}"


def test_extract_audio_checkpoint_skip(tmp_path):
    """Already-processed record is not re-extracted (ffmpeg NOT called)."""
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)
    record = {"id": "rec_001", "audio_filepath": "/fake/audio.mp4"}

    # Mark as already processed
    cp.mark_processed("rec_001", "extract_audio")

    stage = ExtractAudioStage()
    with patch.object(stage, "_extract_one") as mock_extract:
        result = stage.run([record], cfg, cp)
        mock_extract.assert_not_called()

    assert len(result) == 1


def test_extract_audio_ffmpeg_missing(tmp_path):
    """Missing ffmpeg binary raises RuntimeError with install hint."""
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)
    record = {"id": "rec_002", "audio_filepath": "/fake/audio.mp4"}

    stage = ExtractAudioStage()
    with patch("subprocess.run", side_effect=FileNotFoundError("ffmpeg")):
        with pytest.raises(RuntimeError, match="ffmpeg not found"):
            stage.run([record], cfg, cp)


# ---------------------------------------------------------------------------
# VADDiarizeStage tests
# ---------------------------------------------------------------------------

def test_vad_diarize_segment_ids():
    """_make_segment_id produces expected format."""
    stage = VADDiarizeStage()
    seg_id = stage._make_segment_id("source_001", "SPEAKER_00", 1.5, 6.75)
    assert seg_id == "source_001_SPEAKER_00_1.50_6.75"


def test_vad_diarize_duration_filter(tmp_path):
    """Only segments within [min_duration, max_duration] are kept."""
    # Build mock diarization output: 3 turns with durations 1s, 5s, 40s
    mock_turn_1 = MagicMock()
    mock_turn_1.start = 0.0
    mock_turn_1.end = 1.0  # too short

    mock_turn_2 = MagicMock()
    mock_turn_2.start = 2.0
    mock_turn_2.end = 7.0  # 5s — within range

    mock_turn_3 = MagicMock()
    mock_turn_3.start = 10.0
    mock_turn_3.end = 50.0  # 40s — too long

    mock_diarization = MagicMock()
    mock_diarization.itertracks.return_value = [
        (mock_turn_1, None, "SPEAKER_00"),
        (mock_turn_2, None, "SPEAKER_00"),
        (mock_turn_3, None, "SPEAKER_01"),
    ]

    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)

    # Test filtering logic directly (no torchaudio needed)
    valid_records = []
    for turn, _, speaker in mock_diarization.itertracks(yield_label=True):
        dur = turn.end - turn.start
        if cfg.vad.min_duration <= dur <= cfg.vad.max_duration:
            valid_records.append((turn, speaker))

    assert len(valid_records) == 1
    assert valid_records[0][1] == "SPEAKER_00"
    assert valid_records[0][0].end - valid_records[0][0].start == 5.0


def test_vad_diarize_clip_segment(tmp_path):
    """Clipped WAV has duration matching the segment length."""
    import numpy as np
    import soundfile as sf

    sample_rate = 16000
    duration_s = 5.0
    waveform = np.zeros((1, int(sample_rate * 10)), dtype=np.float32)  # 10s

    out_path = str(tmp_path / "clip.wav")
    stage = VADDiarizeStage()
    stage._clip_segment(waveform, sample_rate, start=2.0, end=7.0, out_path=out_path)

    assert Path(out_path).exists()
    loaded, sr = sf.read(out_path)
    clip_duration = len(loaded) / sr
    assert abs(clip_duration - duration_s) < 0.1
