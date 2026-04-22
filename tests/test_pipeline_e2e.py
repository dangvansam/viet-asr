"""
Tests for Phase 6: WriteManifestStage and DataPipeline orchestrator.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from multitalker_asr.data.pipeline.checkpoint import PipelineCheckpoint
from multitalker_asr.data.pipeline.config import PipelineConfig
from multitalker_asr.data.pipeline.pipeline import STAGE_REGISTRY, DataPipeline
from multitalker_asr.data.pipeline.stages.write_manifest import WriteManifestStage


def _make_checkpoint(tmp_path: Path) -> PipelineCheckpoint:
    return PipelineCheckpoint(str(tmp_path / "checkpoints"))


def _make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(output_dir=str(tmp_path))


def _make_record(
    record_id: str = "r1",
    audio: str = "/data/r1.wav",
    text: str = "xin chào",
    text_itn: str = "xin chào .",
    emotion: str = "neutral",
    gender: str = "female",
    language: str = "vi",
    duration: float = 3.5,
    alignment: list = None,
) -> dict:
    return {
        "id": record_id,
        "audio_filepath": audio,
        "text": text,
        "text_itn": text_itn,
        "emotion": emotion,
        "gender": gender,
        "language": language,
        "duration": duration,
        "alignment": alignment or [],
    }


# ---------------------------------------------------------------------------
# WriteManifestStage tests
# ---------------------------------------------------------------------------

def test_write_manifest_skips_empty_text(tmp_path):
    """Record with text='' and text_itn='' is not written to manifest."""
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)

    records = [
        _make_record("r1", text="hello", text_itn="hello ."),
        _make_record("r2", text="", text_itn=""),  # should be skipped
    ]
    stage = WriteManifestStage()
    stage.run(records, cfg, cp)

    manifest = tmp_path / "manifest.jsonl"
    lines = manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["audio_filepath"] == records[0]["audio_filepath"]


def test_write_manifest_textnorm_field(tmp_path):
    """Record with text_itn set → textnorm='withitn'."""
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)

    records = [_make_record("r1", text_itn="xin chào .")]
    stage = WriteManifestStage()
    stage.run(records, cfg, cp)

    manifest = tmp_path / "manifest.jsonl"
    entry = json.loads(manifest.read_text(encoding="utf-8").strip())
    assert entry["textnorm"] == "withitn"


def test_write_manifest_textnorm_none(tmp_path):
    """Record without text_itn → textnorm='none'."""
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)

    record = _make_record("r1", text_itn=None)
    record["text_itn"] = None  # explicitly None
    stage = WriteManifestStage()
    stage.run([record], cfg, cp)

    manifest = tmp_path / "manifest.jsonl"
    entry = json.loads(manifest.read_text(encoding="utf-8").strip())
    assert entry["textnorm"] == "none"


def test_write_manifest_alignment_omitted_if_empty(tmp_path):
    """Empty alignment → 'alignment' key NOT present in manifest entry."""
    cp = _make_checkpoint(tmp_path)
    cfg = _make_config(tmp_path)

    records = [_make_record("r1", alignment=[])]
    stage = WriteManifestStage()
    stage.run(records, cfg, cp)

    manifest = tmp_path / "manifest.jsonl"
    entry = json.loads(manifest.read_text(encoding="utf-8").strip())
    assert "alignment" not in entry


# ---------------------------------------------------------------------------
# DataPipeline tests
# ---------------------------------------------------------------------------

def test_pipeline_stage_registry_complete():
    """All 7 required stage names are in STAGE_REGISTRY."""
    required = {
        "extract_audio",
        "vad_diarize",
        "transcribe",
        "align",
        "gender_classify",
        "enrich_labels",
        "write_manifest",
    }
    assert required.issubset(set(STAGE_REGISTRY.keys()))


def test_pipeline_skip_to_step(tmp_path):
    """pipeline._stages[N:] skips first N stages (--skip_to_step 3 → first 2 removed)."""
    cfg = PipelineConfig(
        output_dir=str(tmp_path),
        stages=["extract_audio", "vad_diarize", "transcribe", "write_manifest"],
        input_dir=str(tmp_path),
    )
    pipeline = DataPipeline(cfg)
    # Simulate --skip_to_step 3
    pipeline._stages = pipeline._stages[2:]
    stage_names = [s.name for s in pipeline._stages]
    assert stage_names == ["transcribe", "write_manifest"]


def test_pipeline_discover_files_respects_max_files(tmp_path):
    """max_files=2 limits discovered files to 2."""
    # Create 4 dummy WAV files
    for i in range(4):
        (tmp_path / f"audio_{i}.wav").touch()

    cfg = PipelineConfig(
        output_dir=str(tmp_path),
        input_dir=str(tmp_path),
        max_files=2,
        stages=["write_manifest"],
    )
    pipeline = DataPipeline(cfg)
    files = pipeline._discover_input_files()
    assert len(files) == 2
