# Phase 3: Transcription Stage — FunASR MLT-Nano

## Goal
Implement `TranscribeStage` that runs FunASR `Fun-ASR-MLT-Nano-2512` on each audio segment,
extracting `text`, `text_itn` (ITN-normalized), `emotion`, and `language` fields.
Also handles pre-transcribed path (`itn_only: true`) — runs FunASR in text-normalization
mode on existing transcripts without re-processing audio.

## Data Flow
```
Input records: {id, audio_filepath, duration, ...}

Mode A — audio transcription (itn_only=false):
    │
    │  FunASR AutoModel("FunAudioLLM/Fun-ASR-MLT-Nano-2512")
    │    .generate(input=audio_filepath, language="auto")
    │    → [{text, raw_text, emotion, language}]
    ▼
Output records: + {text, text_itn, emotion, language}

Mode B — ITN only (itn_only=true, pre-transcribed path):
    │
    │  Input has record["text"] already set (from enrich_labels)
    │  FunASR AutoModel in text_norm mode OR simple rule-based ITN
    │  → normalized text
    ▼
Output records: + {text_itn, emotion (from source or "neutral"), language}

GPU freed after stage: gc.collect() + torch.cuda.empty_cache()
```

## Code Contracts

```python
# src/multitalker_asr/data/pipeline/stages/transcribe.py
import gc
import torch
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
from ..base_stage import BaseStage
from ..config import PipelineConfig, TranscribeConfig
from ..checkpoint import PipelineCheckpoint

class TranscribeStage(BaseStage):
    name = "transcribe"

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        """
        Transcribe audio segments with FunASR MLT-Nano.
        If config.transcribe.itn_only=True: normalize existing text only.
        Frees GPU memory after completing all records.
        """
        ...

    def _load_model(self, cfg: TranscribeConfig):
        """
        Load FunASR AutoModel.
        from funasr import AutoModel
        model = AutoModel(
            model=cfg.model,
            device=cfg.device,
        )
        Cache on self._model — do NOT reload for each record.
        """
        ...

    def _transcribe_audio(
        self,
        audio_path: str,
        model,
        cfg: TranscribeConfig,
    ) -> Dict:
        """
        Call model.generate(input=audio_path, language=cfg.language).
        Parse result list → extract text, raw_text, emotion, language.
        Returns: {"text": str, "text_itn": str, "emotion": str, "language": str}
        FunASR result format:
          [{"text": "normalized text", "raw_text": "raw", "emotion": "<|HAPPY|>", "language": "<|zh|>"}]
        Emotion tag mapping: "<|HAPPY|>"→"happy", "<|SAD|>"→"sad",
          "<|ANGRY|>"→"angry", "<|NEUTRAL|>"→"neutral", None→"neutral"
        Language tag mapping: "<|vi|>"→"vi", "<|zh|>"→"zh", "<|en|>"→"en"
        """
        ...

    def _normalize_text_only(self, text: str, model) -> str:
        """
        Run ITN on existing text string (no audio).
        Use model.generate(input=text, data_type="text") if supported.
        Fallback: return text unchanged if FunASR text mode unavailable.
        """
        ...

    def _free_gpu(self):
        """del self._model; gc.collect(); torch.cuda.empty_cache()"""
        ...
```

## Tasks

### Wave 1

**Task 1a — Implement TranscribeStage (audio mode)**
- **File**: `src/multitalker_asr/data/pipeline/stages/transcribe.py` — new
- **touches**: `[src/multitalker_asr/data/pipeline/stages/transcribe.py]`
- **provides**: `[TranscribeStage]`
- **requires**: `[BaseStage, PipelineConfig, PipelineCheckpoint from Phase 1]`
- **depends_on**: `[Phase 1 complete]`
- Logic for `run()` with `itn_only=False`:
  1. `to_process, done = self._skip_processed(records, checkpoint)`
  2. `self._load_model(config.transcribe)` — loads model ONCE
  3. Iterate `to_process` in batches of `config.transcribe.batch_size`:
     - For each record: call `_transcribe_audio(record["audio_filepath"], model, cfg)`
     - Update record with returned dict fields
     - `checkpoint.mark_processed(record["id"], self.name)`
  4. `checkpoint.save_state()` after each batch
  5. `self._free_gpu()` after all records
  6. Return `done + processed`
- FunASR result parsing:
  - `result = model.generate(input=audio_path, language=language)`
  - `result` is a list; take `result[0]` for single file
  - `text` = `result[0].get("text", "")` — ITN-normalized text
  - `raw_text` = `result[0].get("raw_text", text)` — un-normalized
  - `emotion` = parse emotion tag from text or `result[0].get("emotion", "")`
  - `language` = parse language tag or `result[0].get("language", "vi")`
  - Store: `record["text"] = raw_text`, `record["text_itn"] = text`, `record["emotion"] = emotion_parsed`, `record["language"] = lang_parsed`

**Task 1b — Implement itn_only mode**
- **File**: `src/multitalker_asr/data/pipeline/stages/transcribe.py` — edit Task 1a file
- **touches**: `[src/multitalker_asr/data/pipeline/stages/transcribe.py]`
- **provides**: `[itn_only mode in TranscribeStage]`
- **requires**: `[Task 1a]`
- **depends_on**: `[Task 1a]`
- In `run()`: if `config.transcribe.itn_only`:
  - Skip records that have no `record.get("text")` (log warning)
  - Call `_normalize_text_only(record["text"], model)` → set `record["text_itn"]`
  - Set `record["language"] = record.get("language", "vi")`
  - Set `record["emotion"] = record.get("emotion", "neutral")` (preserve existing if set by enrich_labels)

### Wave 2

**Task 2a — Write tests**
- **File**: `tests/test_pipeline_transcription.py` — new
- **touches**: `[tests/test_pipeline_transcription.py]`
- **provides**: `[transcription tests]`
- **requires**: `[TranscribeStage]`
- **depends_on**: `[Tasks 1a, 1b]`
- Tests:
  - `test_transcribe_emotion_tag_parsing()` — assert `"<|HAPPY|>"` → `"happy"`, `None` → `"neutral"` (no model needed, test parsing function directly)
  - `test_transcribe_language_tag_parsing()` — assert `"<|vi|>"` → `"vi"`, `"<|en|>"` → `"en"`
  - `test_transcribe_itn_only_preserves_emotion()` — records with emotion set in `enrich_labels`; itn_only mode keeps existing emotion
  - `test_transcribe_checkpoint_skip()` — mark record processed, run stage, assert `_transcribe_audio` NOT called
  - `test_transcribe_missing_audio_skipped()` — record with non-existent audio_filepath logged as error and skipped
  - `test_transcribe_gpu_freed()` — after `run()`, assert `self._model` is None (or `_free_gpu` was called)

## Failure Scenarios

| When | Then | Error |
|------|------|-------|
| FunASR not installed | `ImportError` with install hint | `ImportError: funasr not installed — run: pip install funasr` |
| Audio file missing or corrupt | Log error, set `text=""`, continue | `logger.error(f"Transcription failed for {audio_path}: {e}")` |
| FunASR returns empty result list | Set `text=""`, `emotion="neutral"`, `language="vi"` | `logger.warning(f"Empty FunASR result for {audio_path}")` |
| GPU OOM during batch | Reduce batch size to 1, retry once | `logger.warning("OOM — retrying with batch_size=1")` |
| `itn_only=True` but record has no `text` field | Skip record, log warning | `logger.warning(f"itn_only=True but record {id} has no text field")` |
| Model download fails (no internet) | Propagate `OSError` | Do not catch — user must fix connectivity |

## Rejection Criteria (DO NOT)

- DO NOT reload FunASR model for each record — load once, reuse across all records
- DO NOT store the raw FunASR result dict in the output record — parse to standard fields only
- DO NOT assume FunASR result always has `emotion` or `language` keys — use `.get()` with defaults
- DO NOT skip GPU cleanup — `_free_gpu()` MUST be called at end of `run()` (even on exception — use `try/finally`)
- DO NOT call `model.generate()` on batches of files at once if it causes OOM — default to per-file calls

## Cross-Phase Context

**Assumes from Phase 1**:
- `BaseStage`, `PipelineConfig` (with `config.transcribe.*`), `PipelineCheckpoint`

**Assumes from Phase 2 (raw pipeline)**:
- Records have `audio_filepath` pointing to valid 16kHz mono WAV

**Assumes from Phase 5 (pre-transcribed pipeline)**:
- When `itn_only=True`, records already have `text` set by `enrich_labels`
- Records may already have `emotion` set — `TranscribeStage` must NOT overwrite it

**Exports to Phase 4 (align)**:
```python
{
    ...(prior fields),
    "text": str,          # raw transcript (un-normalized)
    "text_itn": str,      # ITN-normalized text (use for manifest)
    "emotion": str,       # "happy" | "sad" | "angry" | "neutral" | "fearful" | "surprised"
    "language": str,      # "vi" | "en" | "zh" | ...
}
```

## Acceptance Criteria

- `uv run pytest tests/test_pipeline_transcription.py` passes (6 tests)
- Emotion tag parsing: `"<|HAPPY|>"` → `"happy"`, missing tag → `"neutral"`
- Language tag parsing: `"<|vi|>"` → `"vi"`, missing → `"vi"` (default)
- GPU freed: `torch.cuda.memory_allocated()` drops after `run()` completes
- `itn_only=True` mode: existing `emotion` field preserved, `text_itn` added
- No FunASR model loaded during import — only inside `_load_model()` (lazy)

## Outcome Block

**What Was Planned**: FunASR MLT-Nano transcription stage with audio and ITN-only modes.
**Immediate Next Action**: Execute Phase 4 — implement `stages/align.py` using Qwen3ForcedAligner.
**How to Measure**:
```bash
uv run pytest tests/test_pipeline_transcription.py -v
python -c "
from src.multitalker_asr.data.pipeline.stages.transcribe import TranscribeStage
s = TranscribeStage()
print('TranscribeStage imported OK, name:', s.name)
"
```
