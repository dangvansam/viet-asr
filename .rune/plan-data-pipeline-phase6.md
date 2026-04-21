# Phase 6: Manifest Writer + Pipeline Orchestrator + CLI

## Goal
Wire everything together: `WriteManifestStage` produces final NeMo JSONL,
`DataPipeline` orchestrates stages in config-defined order, and `scripts/prepare_data.py`
provides the CLI entry point.

## Data Flow
```
scripts/prepare_data.py --config configs/pipeline_raw.yaml \
    --input_dir /data/videos --output_dir /data/processed

    │
    ▼
DataPipeline.run(input_files)
    │
    ├── Load PipelineConfig from YAML
    ├── Apply CLI overrides (input_dir, output_dir, max_files)
    ├── PipelineCheckpoint(config.checkpoint_dir)
    ├── Build stage list from config.stages:
    │     STAGE_REGISTRY = {
    │         "extract_audio":   ExtractAudioStage,
    │         "vad_diarize":     VADDiarizeStage,
    │         "transcribe":      TranscribeStage,
    │         "align":           AlignStage,
    │         "gender_classify": GenderClassifyStage,
    │         "enrich_labels":   EnrichLabelsStage,
    │         "write_manifest":  WriteManifestStage,
    │     }
    ├── Discover input files (glob audio/video from input_dir if raw pipeline)
    │   OR pass [] to EnrichLabelsStage (it reads metadata_path internally)
    ├── Apply max_files limit
    ├── For each stage in order:
    │     records = stage.run(records, config, checkpoint)
    │     logger.success(f"Stage {stage.name}: {len(records)} records")
    └── Return final record count

WriteManifestStage
    │
    │  For each record: build NeMo manifest entry
    │    {
    │      "audio_filepath": record["audio_filepath"],
    │      "text":           record.get("text_itn") or record["text"],
    │      "duration":       record["duration"],
    │      "emotion":        record.get("emotion", "neutral"),
    │      "gender":         record.get("gender", "unknown"),
    │      "language":       record.get("language", "vi"),
    │      "textnorm":       "withitn" if record.get("text_itn") else "none",
    │      "speaker_id":     record.get("speaker_id", "unknown"),
    │    }
    │  Write as JSONL to output_dir/manifest.jsonl
    ▼
Output: NeMo-compatible JSONL manifest
```

## Code Contracts

```python
# src/multitalker_asr/data/pipeline/stages/write_manifest.py
import json
from pathlib import Path
from typing import List, Dict
from loguru import logger
from ..base_stage import BaseStage
from ..config import PipelineConfig
from ..checkpoint import PipelineCheckpoint

class WriteManifestStage(BaseStage):
    name = "write_manifest"

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        """
        Write NeMo JSONL manifest from records.
        Skips records without audio_filepath or text.
        Returns records (unchanged — manifest is a side effect).
        """
        ...

    def _build_manifest_entry(self, record: Dict, cfg) -> Dict:
        """
        Build single NeMo JSONL entry from record.
        Required fields: audio_filepath, text, duration.
        Optional: emotion, gender, language, textnorm, speaker_id, alignment.
        """
        ...
```

```python
# src/multitalker_asr/data/pipeline/pipeline.py
from pathlib import Path
from typing import List, Dict, Optional, Type
from loguru import logger
import gc, torch
from .config import PipelineConfig
from .checkpoint import PipelineCheckpoint
from .base_stage import BaseStage
from .stages.extract_audio import ExtractAudioStage
from .stages.vad_diarize import VADDiarizeStage
from .stages.transcribe import TranscribeStage
from .stages.align import AlignStage
from .stages.gender_classify import GenderClassifyStage
from .stages.enrich_labels import EnrichLabelsStage
from .stages.write_manifest import WriteManifestStage

STAGE_REGISTRY: Dict[str, Type[BaseStage]] = {
    "extract_audio":   ExtractAudioStage,
    "vad_diarize":     VADDiarizeStage,
    "transcribe":      TranscribeStage,
    "align":           AlignStage,
    "gender_classify": GenderClassifyStage,
    "enrich_labels":   EnrichLabelsStage,
    "write_manifest":  WriteManifestStage,
}

class DataPipeline:
    def __init__(self, config: PipelineConfig):
        self._config = config
        self._checkpoint = PipelineCheckpoint(
            config.checkpoint_dir or str(Path(config.output_dir) / "checkpoints")
        )
        self._stages: List[BaseStage] = [
            STAGE_REGISTRY[name]() for name in config.stages
        ]

    def run(self, input_files: Optional[List[str]] = None) -> int:
        """
        Run all stages in order.
        For raw pipeline: input_files is list of audio/video paths.
        For pretranscribed pipeline: input_files is [] (EnrichLabelsStage creates records).
        Returns total manifest entry count.
        """
        ...

    def _discover_input_files(self) -> List[str]:
        """
        Glob audio/video files from config.input_dir.
        Extensions: .wav, .mp3, .mp4, .mkv, .webm, .flac, .m4a
        Apply config.max_files limit if set.
        Returns sorted list of absolute paths.
        """
        ...

    def _make_initial_records(self, files: List[str]) -> List[Dict]:
        """
        Wrap file paths in minimal record dicts for raw pipeline.
        id = Path(f).stem
        {"id": id, "audio_filepath": f, "source_video": f}
        """
        ...

    @classmethod
    def from_yaml(cls, yaml_path: str, **overrides) -> "DataPipeline":
        """
        Load config from YAML, apply kwargs overrides, return DataPipeline.
        Overrides: input_dir, output_dir, max_files, device
        """
        config = PipelineConfig.from_yaml(yaml_path)
        for k, v in overrides.items():
            if v is not None:
                setattr(config, k, v)
        return cls(config)
```

```python
# scripts/prepare_data.py
#!/usr/bin/env python3
"""
Multi-task ASR data processing pipeline CLI.

Usage (raw audio/video):
    uv run python scripts/prepare_data.py \
        --config configs/pipeline_raw.yaml \
        --input_dir /data/tiktok_videos \
        --output_dir /data/processed

Usage (pre-transcribed dataset):
    uv run python scripts/prepare_data.py \
        --config configs/pipeline_pretranscribed.yaml \
        --output_dir /data/processed \
        --enrich_metadata /home/samdv/DATA/asr/emotion_tongdai/transcripts.txt
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from multitalker_asr.data.pipeline.pipeline import DataPipeline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--skip_to_step", type=int, default=None,
                        help="Skip to stage N (1-indexed). All prior stages are skipped.")
    parser.add_argument("--enrich_metadata", default=None,
                        help="Path to pipe-delimited metadata file (pretranscribed pipeline)")
    return parser.parse_args()

def main():
    args = parse_args()
    overrides = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "max_files": args.max_files,
    }
    pipeline = DataPipeline.from_yaml(args.config, **overrides)
    if args.enrich_metadata:
        pipeline._config.enrich.metadata_path = args.enrich_metadata
    if args.skip_to_step:
        pipeline._stages = pipeline._stages[args.skip_to_step - 1:]
    count = pipeline.run()
    print(f"Done. {count} manifest entries written.")

if __name__ == "__main__":
    main()
```

## Tasks

### Wave 1

**Task 1a — Implement WriteManifestStage**
- **File**: `src/multitalker_asr/data/pipeline/stages/write_manifest.py` — new
- **touches**: `[src/multitalker_asr/data/pipeline/stages/write_manifest.py]`
- **provides**: `[WriteManifestStage]`
- **requires**: `[BaseStage, PipelineConfig from Phase 1]`
- **depends_on**: `[Phase 1 complete]`
- `_build_manifest_entry()`:
  - `text_field = config.manifest.text_field` (default: `"text_itn"`)
  - `text = record.get(text_field) or record.get("text", "")`
  - Required: skip records where `text == ""` or `audio_filepath` is falsy
  - Build dict with all 8 fields shown in data flow above
  - Include `"alignment"` only if `record.get("alignment")` is non-empty
- Write JSONL with `ensure_ascii=False`
- Log: `logger.success(f"Manifest written: {manifest_path} ({count} entries)")`

**Task 1b — Implement DataPipeline orchestrator**
- **File**: `src/multitalker_asr/data/pipeline/pipeline.py` — new
- **touches**: `[src/multitalker_asr/data/pipeline/pipeline.py]`
- **provides**: `[DataPipeline, STAGE_REGISTRY]`
- **requires**: `[all stage classes from Phases 2-6 Task 1a, BaseStage, PipelineConfig, PipelineCheckpoint]`
- **depends_on**: `[Tasks from Phase 1; logically depends on all stages being implemented]`
- `run()` logic:
  1. If raw pipeline: `files = self._discover_input_files()`, `records = self._make_initial_records(files)`
  2. If pretranscribed (first stage is `enrich_labels`): `records = []`
  3. For each stage in `self._stages`:
     - `logger.info(f"Running stage: {stage.name} on {len(records)} records")`
     - `records = stage.run(records, self._config, self._checkpoint)`
     - `logger.success(f"Stage {stage.name} complete: {len(records)} records")`
  4. Return `len(records)`
- `_discover_input_files()`: glob with extensions `[".wav", ".mp3", ".mp4", ".mkv", ".webm", ".flac", ".m4a"]`

**Task 1c — Implement scripts/prepare_data.py**
- **File**: `scripts/prepare_data.py` — new
- **touches**: `[scripts/prepare_data.py]`
- **provides**: `[CLI entry point]`
- **requires**: `[DataPipeline]`
- **depends_on**: `[Task 1b]`
- Implement exactly per Code Contracts above

**Task 1d — Update package __init__.py**
- **File**: `src/multitalker_asr/data/pipeline/__init__.py` — edit
- **touches**: `[src/multitalker_asr/data/pipeline/__init__.py]`
- **provides**: `[DataPipeline public export]`
- **requires**: `[DataPipeline from Task 1b]`
- **depends_on**: `[Task 1b]`
- Add: `from .pipeline import DataPipeline`

### Wave 2

**Task 2a — Write tests**
- **File**: `tests/test_pipeline_e2e.py` — new
- **touches**: `[tests/test_pipeline_e2e.py]`
- **provides**: `[e2e tests]`
- **requires**: `[DataPipeline, WriteManifestStage]`
- **depends_on**: `[Tasks 1a, 1b]`
- Tests:
  - `test_write_manifest_skips_empty_text()` — record with `text=""` → not written to manifest
  - `test_write_manifest_textnorm_field()` — record with `text_itn` set → `textnorm="withitn"` in manifest
  - `test_write_manifest_textnorm_none()` — record without `text_itn` → `textnorm="none"`
  - `test_write_manifest_alignment_omitted_if_empty()` — empty alignment → no `alignment` key in manifest entry
  - `test_pipeline_stage_registry_complete()` — assert all 7 stage names in `STAGE_REGISTRY`
  - `test_pipeline_skip_to_step()` — pipeline with `--skip_to_step 3` has first 2 stages removed
  - `test_pipeline_discover_files_respects_max_files()` — max_files=2 → only 2 records created

**Task 2b — Update features.md**
- **File**: `.rune/features.md` — edit
- Update row: `Extended Data Pipeline` status from `Planned` to `Complete`
- Add key files: `data/pipeline/pipeline.py`, `data/pipeline/stages/`

## Failure Scenarios

| When | Then | Error |
|------|------|-------|
| Unknown stage name in YAML | `DataPipeline.__init__` raises `ValueError` | `ValueError: Unknown stage: 'foo'. Valid: {list}` |
| input_dir is empty or missing | `_discover_input_files()` returns [] | `logger.warning("No input files found in {input_dir}")` |
| output_dir not writable | `WriteManifestStage` raises `PermissionError` | Propagate — user must fix permissions |
| All records filtered (no text) | Manifest written with 0 entries | `logger.warning("Manifest has 0 entries — check text extraction")` |
| Stage raises unexpected exception | Log error with traceback, save checkpoint, re-raise | `logger.error(f"Stage {name} failed: {e}")` |

## Rejection Criteria (DO NOT)

- DO NOT hardcode stage order in `DataPipeline` — always read from `config.stages` list
- DO NOT write manifest entries without checking text is non-empty
- DO NOT write `alignment` field when it's an empty list — omit the key entirely
- DO NOT use `json.dumps(ensure_ascii=True)` — Vietnamese text must be Unicode (`ensure_ascii=False`)
- DO NOT call `torch.cuda.empty_cache()` in the orchestrator — each GPU stage handles its own cleanup

## Cross-Phase Context

**Assumes from all prior phases**:
- All stage classes implemented with `BaseStage` interface
- Records evolve with standard fields through stages
- `PipelineCheckpoint` supports resume across all stages

**Final manifest format (NeMo-compatible)**:
```json
{
  "audio_filepath": "/data/segments/seg_001.wav",
  "text": "xin chào các bạn",
  "duration": 3.45,
  "emotion": "neutral",
  "gender": "female",
  "language": "vi",
  "textnorm": "withitn",
  "speaker_id": "SPEAKER_00"
}
```

## Acceptance Criteria

- `uv run pytest tests/test_pipeline_e2e.py` passes (7 tests)
- `uv run python scripts/prepare_data.py --help` runs without error
- Running with `pipeline_raw.yaml` on a test WAV file produces a valid JSONL manifest
- Manifest entries have `ensure_ascii=False` (Vietnamese characters preserved)
- `--skip_to_step 3` skips first 2 stages
- All 6 phases' tests pass: `uv run pytest tests/test_pipeline_*.py`

## Outcome Block

**What Was Planned**: NeMo JSONL manifest writer, DataPipeline orchestrator, and CLI entry point.
**Immediate Next Action**: Execute Phase 1 — implement config dataclasses, checkpoint, and BaseStage.
**How to Measure**:
```bash
# Full test suite
uv run pytest tests/test_pipeline_foundation.py tests/test_pipeline_preprocessing.py \
    tests/test_pipeline_transcription.py tests/test_pipeline_alignment.py \
    tests/test_pipeline_enrichment.py tests/test_pipeline_e2e.py -v

# Smoke test CLI
uv run python scripts/prepare_data.py --help

# Smoke test manifest
uv run python scripts/prepare_data.py \
    --config configs/pipeline_pretranscribed.yaml \
    --output_dir /tmp/test_manifest \
    --enrich_metadata /home/samdv/DATA/tts/nu-mien-bac/metadata.txt \
    --max_files 5
cat /tmp/test_manifest/manifest.jsonl | head -2
```
