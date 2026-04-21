# Phase 1: Foundation — Config + Checkpoint + BaseStage

## Goal
Lay the structural foundation: config dataclasses, crash-resumable checkpoint, and the abstract BaseStage interface that all later stages implement.

## Data Flow
```
YAML file
    │
    ▼
PipelineConfig (dataclass)
    │
    ├── stage_names: List[str]
    ├── input_dir, output_dir
    ├── device, dtype
    └── stage-specific sub-configs (VADConfig, TranscribeConfig, etc.)

PipelineCheckpoint (state manager)
    │
    ├── checkpoints/pipeline_state.json  ← atomic JSON on disk
    ├── is_processed(record_id, stage) → bool
    ├── mark_processed(record_id, stage, metadata)
    └── get_unprocessed(all_ids, stage) → List[str]

BaseStage (abstract)
    │
    └── run(records: List[Dict], config: PipelineConfig, checkpoint: PipelineCheckpoint) → List[Dict]
```

## Code Contracts

```python
# src/multitalker_asr/data/pipeline/config.py
from dataclasses import dataclass, field
from typing import List, Optional
from ..configs import BaseConfig  # existing base

@dataclass
class VADConfig:
    min_duration: float = 3.0
    max_duration: float = 30.0
    model: str = "pyannote-onnx"
    hf_token: Optional[str] = None

@dataclass
class TranscribeConfig:
    model: str = "FunAudioLLM/Fun-ASR-MLT-Nano-2512"
    language: str = "auto"
    output_itn: bool = True
    itn_only: bool = False   # text-only normalization (pre-transcribed path)
    device: str = "cuda"
    batch_size: int = 8

@dataclass
class AlignConfig:
    model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    language: str = "Vietnamese"
    device: str = "cuda"
    dtype: str = "bfloat16"

@dataclass
class GenderConfig:
    url: str = "http://localhost:8000/predict"
    model: str = "ensemble"
    timeout: int = 30

@dataclass
class EnrichConfig:
    input_format: str = "pipe_delimited"   # "pipe_delimited" only for now
    # label_mapping: maps source labels → standard labels
    # e.g. {"NEU": "neutral", "POS": "positive", "NEG": "negative"}
    emotion_mapping: dict = field(default_factory=lambda: {
        "NEU": "neutral", "POS": "positive", "NEG": "negative"
    })

@dataclass
class ManifestConfig:
    output_filename: str = "manifest.jsonl"
    text_field: str = "text_itn"   # "text_itn" or "text"

@dataclass
class PipelineConfig(BaseConfig):
    pipeline_name: str = "raw"          # "raw" or "pretranscribed"
    stages: List[str] = field(default_factory=list)
    input_dir: str = ""
    output_dir: str = ""
    checkpoint_dir: str = ""            # defaults to output_dir/checkpoints
    max_files: Optional[int] = None     # None = process all
    device: str = "cuda"
    vad: VADConfig = field(default_factory=VADConfig)
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    gender: GenderConfig = field(default_factory=GenderConfig)
    enrich: EnrichConfig = field(default_factory=EnrichConfig)
    manifest: ManifestConfig = field(default_factory=ManifestConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig": ...
    # loads YAML, constructs nested dataclasses via OmegaConf or manual dict merge
```

```python
# src/multitalker_asr/data/pipeline/checkpoint.py
# COPY from /home/samdv/data-processing-pipeline/pipeline/checkpoint.py
# CHANGE: replace print() calls with loguru logger.info/warning/error
# KEEP: all method signatures identical (is_processed, mark_processed,
#        mark_batch_processed, get_unprocessed_files, get_step_statistics,
#        clear_step, clear_all, get_progress_summary)
class PipelineCheckpoint: ...   # full copy, logging only change
```

```python
# src/multitalker_asr/data/pipeline/base_stage.py
from abc import ABC, abstractmethod
from typing import List, Dict
from .config import PipelineConfig
from .checkpoint import PipelineCheckpoint

class BaseStage(ABC):
    name: str  # class-level, matches stage key in YAML stages list

    @abstractmethod
    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        """Process records, skip already-processed, return enriched records."""
        ...

    def _skip_processed(
        self, records: List[Dict], checkpoint: PipelineCheckpoint
    ) -> tuple[List[Dict], List[Dict]]:
        """Split records into (to_process, already_done)."""
        to_process, done = [], []
        for r in records:
            if checkpoint.is_processed(r["id"], self.name):
                done.append(r)
            else:
                to_process.append(r)
        return to_process, done
```

## Tasks

### Wave 1 (independent, can be written in parallel)

**Task 1a — Create package skeleton**
- **File**: `src/multitalker_asr/data/pipeline/__init__.py` — new
- **touches**: `[src/multitalker_asr/data/pipeline/__init__.py]`
- **provides**: `[pipeline package]`
- **requires**: []
- Content: `from .pipeline import DataPipeline` (forward ref only — `DataPipeline` added in Phase 6)
- For now: empty `__init__.py` is fine; add export in Phase 6

**Task 1b — Create stages package**
- **File**: `src/multitalker_asr/data/pipeline/stages/__init__.py` — new
- **touches**: `[src/multitalker_asr/data/pipeline/stages/__init__.py]`
- **provides**: `[stages package]`
- **requires**: []
- Content: empty file

**Task 1c — Write config.py**
- **File**: `src/multitalker_asr/data/pipeline/config.py` — new
- **touches**: `[src/multitalker_asr/data/pipeline/config.py]`
- **provides**: `[PipelineConfig, VADConfig, TranscribeConfig, AlignConfig, GenderConfig, EnrichConfig, ManifestConfig]`
- **requires**: `[BaseConfig from src/multitalker_asr/configs/__init__.py]`
- Implement all dataclasses from Code Contracts above
- `PipelineConfig.from_yaml()`: load YAML with `OmegaConf.load()`, convert to dict, construct nested dataclasses
- Edge case: if `checkpoint_dir` is empty string, default to `output_dir + "/checkpoints"`

**Task 1d — Write checkpoint.py**
- **File**: `src/multitalker_asr/data/pipeline/checkpoint.py` — new
- **touches**: `[src/multitalker_asr/data/pipeline/checkpoint.py]`
- **provides**: `[PipelineCheckpoint]`
- **requires**: []
- Copy `/home/samdv/data-processing-pipeline/pipeline/checkpoint.py` verbatim
- Replace all `print(...)` calls with `logger.info/warning/error` from loguru
- DO NOT change any method signatures or logic

**Task 1e — Write base_stage.py**
- **File**: `src/multitalker_asr/data/pipeline/base_stage.py` — new
- **touches**: `[src/multitalker_asr/data/pipeline/base_stage.py]`
- **provides**: `[BaseStage]`
- **requires**: `[PipelineConfig, PipelineCheckpoint]`
- Implement `BaseStage` ABC from Code Contracts above

### Wave 2 (after Wave 1)

**Task 2a — Write pipeline_raw.yaml**
- **File**: `configs/pipeline_raw.yaml` — new
- **touches**: `[configs/pipeline_raw.yaml]`
- **provides**: `[raw pipeline config]`
- **requires**: `[PipelineConfig schema from Task 1c]`
- Content:
```yaml
pipeline_name: raw
stages: [extract_audio, vad_diarize, transcribe, align, gender_classify, write_manifest]
input_dir: ""        # override at CLI with --input_dir
output_dir: ""       # override at CLI with --output_dir
device: cuda
vad:
  min_duration: 3.0
  max_duration: 30.0
  model: pyannote-onnx
transcribe:
  model: FunAudioLLM/Fun-ASR-MLT-Nano-2512
  language: auto
  output_itn: true
  itn_only: false
align:
  model: Qwen/Qwen3-ForcedAligner-0.6B
  language: Vietnamese
  dtype: bfloat16
gender:
  url: http://localhost:8000/predict
  model: ensemble
manifest:
  output_filename: manifest.jsonl
  text_field: text_itn
```

**Task 2b — Write pipeline_pretranscribed.yaml**
- **File**: `configs/pipeline_pretranscribed.yaml` — new
- **touches**: `[configs/pipeline_pretranscribed.yaml]`
- **provides**: `[pretranscribed pipeline config]`
- **requires**: `[PipelineConfig schema from Task 1c]`
- Content: same as raw but:
  - `stages: [enrich_labels, transcribe, gender_classify, write_manifest]`
  - `transcribe.itn_only: true`
  - `enrich.input_format: pipe_delimited`

**Task 2c — Write tests**
- **File**: `tests/test_pipeline_foundation.py` — new
- **touches**: `[tests/test_pipeline_foundation.py]`
- **provides**: `[foundation tests]`
- **requires**: `[Tasks 1c, 1d, 1e]`
- Tests:
  - `test_pipeline_config_from_yaml()` — load both YAML configs, assert stage lists correct
  - `test_checkpoint_mark_and_check()` — mark file processed, assert is_processed returns True
  - `test_checkpoint_resume()` — mark 3 of 5 files, call get_unprocessed_files, assert 2 returned
  - `test_checkpoint_atomic_save()` — verify temp+rename pattern (check .json.tmp not left behind)
  - `test_base_stage_skip_processed()` — create concrete BaseStage stub, verify _skip_processed splits correctly

## Failure Scenarios

| When | Then | Error |
|------|------|-------|
| YAML file missing | `from_yaml()` raises `FileNotFoundError` | `FileNotFoundError: Config not found: path` |
| YAML has unknown key | Log warning, ignore key (OmegaConf struct=False) | `logger.warning("Unknown config key: X")` |
| checkpoint_dir not writable | `PipelineCheckpoint.__init__` raises `PermissionError` | Propagate, let CLI print helpful message |
| checkpoint JSON corrupted | `load_state()` catches exception, starts with empty state | `logger.warning("Failed to load checkpoint, starting fresh")` |
| `BaseConfig` import fails (NeMo not installed) | Fall back to plain `object` as base | Add `try/except ImportError` around base import |

## Rejection Criteria (DO NOT)

- DO NOT use stdlib `logging` — always `from loguru import logger`
- DO NOT use `@dataclass(frozen=True)` — configs must be mutable for CLI overrides
- DO NOT put any model loading code in config or checkpoint files
- DO NOT store absolute file paths in checkpoint JSON — store relative to `input_dir` or use record `id` field
- DO NOT change checkpoint method signatures from the reference implementation

## Cross-Phase Context

**Assumes**: `src/multitalker_asr/configs/__init__.py` exports `BaseConfig` (verified existing)

**Exports to all later phases**:
- `PipelineConfig` — passed to every stage's `run()` method
- `PipelineCheckpoint` — passed to every stage's `run()` method
- `BaseStage` — every stage in Phases 2–6 inherits from this
- Record schema: every stage receives `List[Dict]` and returns `List[Dict]`; minimum keys: `{"id": str, "audio_filepath": str}`

## Acceptance Criteria

- `uv run pytest tests/test_pipeline_foundation.py` passes (5 tests)
- `PipelineConfig.from_yaml("configs/pipeline_raw.yaml")` returns config with `stages == ["extract_audio", "vad_diarize", "transcribe", "align", "gender_classify", "write_manifest"]`
- Checkpoint: after marking 3 of 5 records processed, `get_unprocessed_files` returns exactly 2
- Checkpoint atomic save: `.json.tmp` file is never left behind after `save_state()`
- No `print()` statements in checkpoint.py — all output via loguru

## Outcome Block

**What Was Planned**: Config dataclasses, crash-resumable checkpoint, and BaseStage interface.
**Immediate Next Action**: Execute Phase 2 — implement `extract_audio.py` and `vad_diarize.py`.
**How to Measure**:
```bash
uv run pytest tests/test_pipeline_foundation.py -v
python -c "from src.multitalker_asr.data.pipeline.config import PipelineConfig; c = PipelineConfig.from_yaml('configs/pipeline_raw.yaml'); print(c.stages)"
```
