"""
Tests for Phase 1: Pipeline foundation — config, checkpoint, base_stage.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

from multitalker_asr.data.pipeline.base_stage import BaseStage
from multitalker_asr.data.pipeline.checkpoint import PipelineCheckpoint
from multitalker_asr.data.pipeline.config import PipelineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_checkpoint(tmp_path: Path) -> PipelineCheckpoint:
    return PipelineCheckpoint(str(tmp_path / "checkpoints"))


# ---------------------------------------------------------------------------
# PipelineConfig tests
# ---------------------------------------------------------------------------

def test_pipeline_config_from_yaml_raw(tmp_path):
    """Loading pipeline_raw.yaml returns config with correct stage list."""
    yaml_path = Path(__file__).parent.parent / "configs" / "pipeline_raw.yaml"
    assert yaml_path.exists(), f"YAML missing: {yaml_path}"

    cfg = PipelineConfig.from_yaml(str(yaml_path))
    assert cfg.stages == [
        "extract_audio",
        "vad_diarize",
        "transcribe",
        "align",
        "gender_classify",
        "write_manifest",
    ]
    assert cfg.pipeline_name == "raw"


def test_pipeline_config_from_yaml_pretranscribed(tmp_path):
    """Loading pipeline_pretranscribed.yaml returns config with correct stage list."""
    yaml_path = Path(__file__).parent.parent / "configs" / "pipeline_pretranscribed.yaml"
    assert yaml_path.exists(), f"YAML missing: {yaml_path}"

    cfg = PipelineConfig.from_yaml(str(yaml_path))
    assert cfg.stages == [
        "enrich_labels",
        "transcribe",
        "gender_classify",
        "write_manifest",
    ]
    assert cfg.transcribe.itn_only is True


def test_pipeline_config_from_yaml_missing_file():
    """from_yaml raises FileNotFoundError for non-existent path."""
    with pytest.raises(FileNotFoundError, match="Config not found"):
        PipelineConfig.from_yaml("/nonexistent/path/config.yaml")


def test_pipeline_config_checkpoint_dir_default(tmp_path):
    """checkpoint_dir defaults to output_dir/checkpoints when not set."""
    cfg = PipelineConfig(output_dir=str(tmp_path))
    assert cfg.checkpoint_dir == str(tmp_path / "checkpoints")


# ---------------------------------------------------------------------------
# PipelineCheckpoint tests
# ---------------------------------------------------------------------------

def test_checkpoint_mark_and_check(tmp_path):
    """After mark_processed, is_processed returns True."""
    cp = _make_checkpoint(tmp_path)
    cp.mark_processed("file_001", "extract_audio")
    assert cp.is_processed("file_001", "extract_audio") is True
    assert cp.is_processed("file_002", "extract_audio") is False


def test_checkpoint_resume(tmp_path):
    """Mark 3 of 5 records processed; get_unprocessed_files returns exactly 2."""
    cp = _make_checkpoint(tmp_path)
    all_files = ["a", "b", "c", "d", "e"]
    for f in all_files[:3]:
        cp.mark_processed(f, "vad_diarize")

    unprocessed = cp.get_unprocessed_files(all_files, "vad_diarize")
    assert sorted(unprocessed) == ["d", "e"]


def test_checkpoint_atomic_save(tmp_path):
    """After save_state(), no .json.tmp file is left behind."""
    cp = _make_checkpoint(tmp_path)
    cp.mark_processed("rec_01", "transcribe")
    cp.save_state()

    tmp_file = cp.checkpoint_dir / "pipeline_state.json.tmp"
    assert not tmp_file.exists(), ".json.tmp should not exist after save_state()"
    assert cp.checkpoint_file.exists(), "checkpoint JSON must exist after save"


def test_checkpoint_corrupted_json_starts_fresh(tmp_path):
    """Corrupted checkpoint JSON causes fresh state, no exception raised."""
    cp_dir = tmp_path / "checkpoints"
    cp_dir.mkdir()
    state_file = cp_dir / "pipeline_state.json"
    state_file.write_text("{not valid json", encoding="utf-8")

    # Should not raise, should start fresh
    cp = PipelineCheckpoint(str(cp_dir))
    assert cp.state == {}


# ---------------------------------------------------------------------------
# BaseStage tests
# ---------------------------------------------------------------------------

class _ConcreteStage(BaseStage):
    """Minimal concrete stage for testing _skip_processed."""

    name = "test_stage"

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        to_process, done = self._skip_processed(records, checkpoint)
        return done + to_process  # just return all, no-op


def test_base_stage_skip_processed(tmp_path):
    """_skip_processed correctly splits records into done and to_process."""
    cp = _make_checkpoint(tmp_path)
    records = [{"id": "r1", "audio_filepath": "a.wav"},
               {"id": "r2", "audio_filepath": "b.wav"},
               {"id": "r3", "audio_filepath": "c.wav"}]

    cp.mark_processed("r1", "test_stage")
    cp.mark_processed("r3", "test_stage")

    stage = _ConcreteStage()
    to_process, done = stage._skip_processed(records, cp)

    assert len(done) == 2
    assert len(to_process) == 1
    assert to_process[0]["id"] == "r2"
    done_ids = {r["id"] for r in done}
    assert done_ids == {"r1", "r3"}
