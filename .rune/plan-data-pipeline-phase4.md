# Phase 4: Alignment Stage — Qwen3ForcedAligner

## Goal
Implement `AlignStage` that runs `Qwen3ForcedAligner-0.6B` to produce word-level timestamps
for each audio segment. Adapted directly from the proven implementation at
`/home/samdv/data-processing-pipeline/pipeline/alignment.py`.

## Data Flow
```
Input records: {id, audio_filepath, text, text_itn, emotion, language, ...}

AlignStage
    │
    │  from qwen_asr import Qwen3ForcedAligner
    │  aligner = Qwen3ForcedAligner.from_pretrained(model, dtype=bfloat16)
    │
    │  For each record:
    │    results = aligner.align(
    │        audio=[record["audio_filepath"]],
    │        text=[record["text_itn"]],   ← use ITN text for alignment
    │        language=["Vietnamese"],
    │    )
    │    → results[0] = List[ForcedAlignResult(text, start_time, end_time)]
    ▼
Output records: + {
    "alignment": [{"text": str, "start_time": float, "end_time": float}, ...],
    "alignment_score": float   # fraction of valid timestamps (0.0–1.0)
}

GPU freed after stage: gc.collect() + torch.cuda.empty_cache()
```

## Code Contracts

```python
# src/multitalker_asr/data/pipeline/stages/align.py
# Adapted from /home/samdv/data-processing-pipeline/pipeline/alignment.py
# Key changes:
#   - Wrapped in BaseStage interface
#   - Uses PipelineConfig/PipelineCheckpoint instead of raw function args
#   - Uses loguru instead of print()
#   - Adds _skip_processed checkpoint logic

import gc
import torch
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
from ..base_stage import BaseStage
from ..config import PipelineConfig, AlignConfig
from ..checkpoint import PipelineCheckpoint

class AlignStage(BaseStage):
    name = "align"

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        """
        Run Qwen3ForcedAligner on each record.
        Uses record["text_itn"] as input text for alignment.
        Falls back to record["text"] if text_itn missing.
        Frees GPU after all records processed.
        """
        ...

    def _load_aligner(self, cfg: AlignConfig):
        """
        from qwen_asr import Qwen3ForcedAligner
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        aligner = Qwen3ForcedAligner.from_pretrained(
            cfg.model,
            dtype=dtype_map[cfg.dtype],
            device_map=cfg.device + ":0",
        )
        Store as self._aligner.
        """
        ...

    def _align_one(
        self,
        audio_path: str,
        text: str,
        language: str,
    ) -> List[Dict]:
        """
        results = self._aligner.align(
            audio=[audio_path],
            text=[text],
            language=[language],
        )
        Parse results[0] → List[{"text": str, "start_time": float, "end_time": float}]
        Returns empty list on failure (do not raise).
        """
        ...

    def _calculate_alignment_score(self, alignment: List[Dict]) -> float:
        """
        Fraction of segments with valid timestamps (start >= 0 and end > start).
        Returns 0.0 for empty alignment.
        """
        ...

    def _free_gpu(self):
        """del self._aligner; gc.collect(); torch.cuda.empty_cache()"""
        ...
```

## Tasks

### Wave 1

**Task 1a — Implement AlignStage**
- **File**: `src/multitalker_asr/data/pipeline/stages/align.py` — new
- **touches**: `[src/multitalker_asr/data/pipeline/stages/align.py]`
- **provides**: `[AlignStage]`
- **requires**: `[BaseStage, PipelineConfig, PipelineCheckpoint from Phase 1]`
- **depends_on**: `[Phase 1 complete]`
- Reference: copy core logic from `/home/samdv/data-processing-pipeline/pipeline/alignment.py`
  - Keep: `aligner.align(audio=[...], text=[...], language=[...])` call pattern
  - Keep: result parsing loop `for item in results[0]`
  - Keep: `_calculate_alignment_score()` (identical to reference `calculate_alignment_score()`)
  - Change: wrap in `BaseStage.run()` with checkpoint skip
  - Change: `print()` → `logger.info/warning/error`
  - Change: use `record["text_itn"] or record.get("text", "")` as input text
- Logic for `run()`:
  1. `to_process, done = self._skip_processed(records, checkpoint)`
  2. `self._load_aligner(config.align)` — loads ONCE before loop
  3. For each record in `to_process`:
     - `text = record.get("text_itn") or record.get("text", "")`
     - `language = config.align.language`
     - If text is empty: set `record["alignment"] = []`, `record["alignment_score"] = 0.0`, skip align call
     - Else: call `_align_one(record["audio_filepath"], text, language)`
     - Set `record["alignment"]` and `record["alignment_score"]`
     - `checkpoint.mark_processed(record["id"], self.name)`
     - `checkpoint.save_state()` after each record (alignment is slow — save often)
  4. `self._free_gpu()` in `finally` block

### Wave 2

**Task 2a — Write tests**
- **File**: `tests/test_pipeline_alignment.py` — new
- **touches**: `[tests/test_pipeline_alignment.py]`
- **provides**: `[alignment tests]`
- **requires**: `[AlignStage]`
- **depends_on**: `[Task 1a]`
- Tests:
  - `test_align_score_valid()` — alignment with all valid timestamps → score == 1.0
  - `test_align_score_partial()` — 2 of 4 timestamps valid → score == 0.5
  - `test_align_score_empty()` — empty alignment → score == 0.0
  - `test_align_empty_text_skipped()` — record with empty text → alignment=[], score=0.0, no aligner call
  - `test_align_checkpoint_skip()` — mark record processed, run stage, assert `_align_one` NOT called
  - `test_align_fallback_to_text()` — record with `text_itn=None` uses `text` field as fallback
  - `test_align_gpu_freed()` — after `run()`, `self._aligner` should be None

## Failure Scenarios

| When | Then | Error |
|------|------|-------|
| `qwen_asr` not installed | ImportError with install hint | `ImportError: qwen_asr not installed` |
| Model not downloaded | OSError on `from_pretrained` | Propagate — user must have internet/HF access |
| Alignment returns empty results | Set `alignment=[]`, `score=0.0`, continue | `logger.warning(f"Empty alignment for {audio_path}")` |
| Alignment raises exception | Log error, set `alignment=[]`, `score=0.0`, continue | `logger.error(f"Alignment failed: {e}")` |
| GPU OOM on 30s segment | Log error, skip alignment for that record | `logger.error("OOM during alignment for {id}")` |
| `text_itn` and `text` both empty | Skip alignment, set empty fields | `logger.warning(f"No text for alignment: {id}")` |

## Rejection Criteria (DO NOT)

- DO NOT reload `Qwen3ForcedAligner` for each record — load ONCE per stage execution
- DO NOT use `record["text_itn"]` without `.get()` fallback — key may not exist in pre-transcribed path
- DO NOT call `aligner.align()` with `text=""` — check text non-empty before calling
- DO NOT omit `_free_gpu()` in a `finally` block — GPU must be freed even if an exception occurs
- DO NOT batch multiple files into one `aligner.align()` call unless tested — reference impl does per-file calls

## Cross-Phase Context

**Assumes from Phase 1**: `BaseStage`, `PipelineConfig` (with `config.align.*`), `PipelineCheckpoint`

**Assumes from Phase 3**:
- Records have `text_itn` (primary) or `text` (fallback) set
- Records have `audio_filepath` pointing to valid WAV

**Exports to Phase 5 (gender_classify)**:
```python
{
    ...(prior fields),
    "alignment": [
        {"text": str, "start_time": float, "end_time": float},
        ...
    ],
    "alignment_score": float,   # 0.0–1.0
}
```

Note: `alignment` field is OPTIONAL in the final manifest — write_manifest skips it if empty.

## Acceptance Criteria

- `uv run pytest tests/test_pipeline_alignment.py` passes (7 tests)
- `_calculate_alignment_score([{"start_time": 0.1, "end_time": 0.5}])` == 1.0
- `_calculate_alignment_score([])` == 0.0
- Empty text record: `alignment == []`, `alignment_score == 0.0`, no aligner loaded
- GPU freed: after `run()`, `torch.cuda.memory_allocated()` returns to pre-run level
- `checkpoint.save_state()` called after EACH record (not just at end of batch)

## Outcome Block

**What Was Planned**: Qwen3ForcedAligner-based word-level timestamp alignment stage.
**Immediate Next Action**: Execute Phase 5 — implement `stages/gender_classify.py` and `stages/enrich_labels.py`.
**How to Measure**:
```bash
uv run pytest tests/test_pipeline_alignment.py -v
python -c "
from src.multitalker_asr.data.pipeline.stages.align import AlignStage
s = AlignStage()
print('AlignStage name:', s.name)
print('score test:', s._calculate_alignment_score([{'start_time': 0.1, 'end_time': 0.5}]))
"
```
