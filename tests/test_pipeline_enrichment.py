"""
Tests for Phase 5: GenderClassifyStage and EnrichLabelsStage.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from multitalker_asr.data.pipeline.checkpoint import PipelineCheckpoint
from multitalker_asr.data.pipeline.config import PipelineConfig
from multitalker_asr.data.pipeline.stages.gender_classify import GenderClassifyStage
from multitalker_asr.data.pipeline.stages.enrich_labels import EnrichLabelsStage


def _make_checkpoint(tmp_path: Path) -> PipelineCheckpoint:
    return PipelineCheckpoint(str(tmp_path / "checkpoints"))


def _make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(output_dir=str(tmp_path))


def _mock_response(gender: str = "MALE", probs=None):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"gender": gender, "probs": probs or [0.2, 0.8]}
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# GenderClassifyStage tests
# ---------------------------------------------------------------------------

def test_gender_classify_lowercases(tmp_path):
    """HTTP response with 'MALE' → record has 'male'."""
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)

    # Create a real (empty) wav file so open() doesn't fail
    fake_wav = tmp_path / "r1.wav"
    fake_wav.touch()

    records = [{"id": "r1", "audio_filepath": str(fake_wav)}]
    stage = GenderClassifyStage()

    with patch.object(stage, "_check_service", return_value=True), \
         patch("requests.post", return_value=_mock_response("MALE", [0.2, 0.8])):
        result = stage.run(records, cfg, cp)

    assert result[0]["gender"] == "male"
    assert result[0]["gender_confidence"] == 0.8


def test_gender_classify_skips_existing(tmp_path):
    """Record with gender already set does not trigger HTTP call."""
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)

    records = [{"id": "r1", "audio_filepath": "/fake.wav", "gender": "female"}]
    stage = GenderClassifyStage()

    with patch.object(stage, "_check_service", return_value=True), \
         patch("requests.post") as mock_post:
        result = stage.run(records, cfg, cp)
        mock_post.assert_not_called()

    assert result[0]["gender"] == "female"


def test_gender_service_down_skips_stage(tmp_path):
    """When service is unreachable, records are returned unchanged."""
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)

    records = [{"id": "r1", "audio_filepath": "/fake.wav"}]
    stage = GenderClassifyStage()

    with patch.object(stage, "_check_service", return_value=False), \
         patch("requests.post") as mock_post:
        result = stage.run(records, cfg, cp)
        mock_post.assert_not_called()

    # Gender field should not be modified
    assert result[0].get("gender") is None


def test_gender_classify_checkpoint_skip(tmp_path):
    """Already-processed record does not trigger HTTP call."""
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)
    cp.mark_processed("r1", "gender_classify")

    records = [{"id": "r1", "audio_filepath": "/fake.wav"}]
    stage = GenderClassifyStage()

    with patch.object(stage, "_check_service", return_value=True), \
         patch("requests.post") as mock_post:
        result = stage.run(records, cfg, cp)
        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# EnrichLabelsStage tests
# ---------------------------------------------------------------------------

def test_enrich_nu_mien_bac_format(tmp_path):
    """4-field format: gender='female', emotion='neutral'."""
    metadata = tmp_path / "metadata.txt"
    metadata.write_text(
        "001|nu-mien-bac|/data/001.wav|xin chào các bạn\n",
        encoding="utf-8",
    )

    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)
    cfg.enrich.metadata_path = str(metadata)

    stage = EnrichLabelsStage()
    records = stage.run([], cfg, cp)

    assert len(records) == 1
    assert records[0]["gender"] == "female"
    assert records[0]["emotion"] == "neutral"
    assert records[0]["text"] == "xin chào các bạn"
    assert records[0]["language"] == "vi"


def test_enrich_emotion_tongdai_format(tmp_path):
    """5-field format: emotion mapped, duration parsed as float."""
    metadata = tmp_path / "metadata.txt"
    metadata.write_text(
        "001|/data/001.wav|NEG|3.5|tôi không thích điều đó\n",
        encoding="utf-8",
    )

    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)
    cfg.enrich.metadata_path = str(metadata)

    stage = EnrichLabelsStage()
    records = stage.run([], cfg, cp)

    assert records[0]["emotion"] == "negative"
    assert records[0]["duration"] == pytest.approx(3.5)
    assert records[0]["gender"] is None


def test_enrich_unknown_emotion_defaults_neutral(tmp_path):
    """Unknown emotion tag defaults to 'neutral'."""
    metadata = tmp_path / "metadata.txt"
    metadata.write_text(
        "001|/data/001.wav|WEIRD|2.0|some text\n",
        encoding="utf-8",
    )

    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)
    cfg.enrich.metadata_path = str(metadata)

    stage = EnrichLabelsStage()
    records = stage.run([], cfg, cp)

    assert records[0]["emotion"] == "neutral"
