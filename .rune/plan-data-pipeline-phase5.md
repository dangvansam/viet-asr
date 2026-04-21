# Phase 5: Enrichment Stages — gender_classify + enrich_labels

## Goal
Two stages:
1. `GenderClassifyStage` — calls gender-classification-service HTTP API per audio segment
2. `EnrichLabelsStage` — parses pipe-delimited source metadata (nu-mien-bac, emotion_tongdai)
   and maps existing labels into standard record fields before transcription

## Data Flow

### GenderClassifyStage
```
Input records: {id, audio_filepath, ...}
    │
    │  HTTP POST http://localhost:8000/predict
    │    files={"audiofile": open(audio_filepath, "rb")}
    │    params={"model": "ensemble", "return_label": False}
    │  → {"gender": "MALE"|"FEMALE", "probs": [female_p, male_p], ...}
    ▼
Output records: + {
    "gender": "male" | "female",        # lowercased
    "gender_confidence": float          # max(probs[0], probs[1])
}
```

### EnrichLabelsStage (pre-transcribed pipeline only, runs FIRST)
```
Input: raw metadata file path (from config)

Format A — nu-mien-bac:
    {id}|{speaker}|{wav_path}|{text}
    → gender = "female" (inferred from speaker name "nu-mien-bac")
    → emotion = "neutral" (TTS, no emotion label)

Format B — emotion_tongdai:
    {id}|{wav_path}|{emotion}|{duration}|{text}
    → emotion mapped via config.enrich.emotion_mapping (NEU→neutral, POS→positive, NEG→negative)
    → gender = None (will be classified in GenderClassifyStage)

Output records (one per metadata line):
{
    "id": str,               # line id or generated hash
    "audio_filepath": str,   # absolute path to WAV
    "text": str,             # original transcript
    "text_itn": None,        # filled by TranscribeStage (itn_only mode)
    "emotion": str,          # mapped label or "neutral"
    "gender": str | None,    # known from source or None
    "language": "vi",        # assumed Vietnamese
    "duration": float | None # from metadata if available, else None
}
```

## Code Contracts

```python
# src/multitalker_asr/data/pipeline/stages/gender_classify.py
import requests
from pathlib import Path
from typing import List, Dict
from loguru import logger
from ..base_stage import BaseStage
from ..config import PipelineConfig, GenderConfig
from ..checkpoint import PipelineCheckpoint

class GenderClassifyStage(BaseStage):
    name = "gender_classify"

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        """
        Classify gender for each audio segment via HTTP API.
        Skips records already processed (checkpoint).
        Skips records where gender is already set (from enrich_labels).
        """
        ...

    def _classify_one(self, audio_path: str, cfg: GenderConfig) -> Dict:
        """
        POST to cfg.url with audio file.
        Returns {"gender": "male"|"female", "gender_confidence": float}
        Raises requests.Timeout if service not responding within cfg.timeout seconds.
        """
        ...

    def _check_service(self, url: str, timeout: int = 5) -> bool:
        """GET {url}/health_check → True if 200, False otherwise."""
        ...
```

```python
# src/multitalker_asr/data/pipeline/stages/enrich_labels.py
from pathlib import Path
from typing import List, Dict
from loguru import logger
from ..base_stage import BaseStage
from ..config import PipelineConfig, EnrichConfig
from ..checkpoint import PipelineCheckpoint

class EnrichLabelsStage(BaseStage):
    name = "enrich_labels"

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        """
        Parse pipe-delimited metadata file and return initial records.
        'records' input is typically empty [] (this stage creates records from scratch).
        Uses config.enrich.input_format to determine parsing logic.
        """
        ...

    def _parse_metadata_file(
        self,
        metadata_path: str,
        cfg: EnrichConfig,
    ) -> List[Dict]:
        """
        Read metadata_path line by line.
        Detect format by counting pipe '|' separators per line.
        4 fields → nu_mien_bac format: id|speaker|wav_path|text
        5 fields → emotion_tongdai format: id|wav_path|emotion|duration|text
        Generate record id as: Path(wav_path).stem
        """
        ...

    def _make_record_id(self, wav_path: str) -> str:
        """Return Path(wav_path).stem as record id."""
        ...
```

## Tasks

### Wave 1 (independent — both stages can be written in parallel)

**Task 1a — Implement GenderClassifyStage**
- **File**: `src/multitalker_asr/data/pipeline/stages/gender_classify.py` — new
- **touches**: `[src/multitalker_asr/data/pipeline/stages/gender_classify.py]`
- **provides**: `[GenderClassifyStage]`
- **requires**: `[BaseStage, PipelineConfig, PipelineCheckpoint from Phase 1]`
- **depends_on**: `[Phase 1 complete]`
- Logic for `run()`:
  1. `_check_service(config.gender.url)` — if fails, log warning and skip stage (return records unchanged)
  2. `to_process, done = self._skip_processed(records, checkpoint)`
  3. Also skip records where `record.get("gender")` is already set (from enrich_labels)
  4. For each record:
     - `result = _classify_one(record["audio_filepath"], config.gender)`
     - `record["gender"] = result["gender"]`
     - `record["gender_confidence"] = result["gender_confidence"]`
     - `checkpoint.mark_processed(record["id"], self.name)`
  5. `checkpoint.save_state()` every 50 records
  6. Return `done + processed`
- `_classify_one()` implementation:
  ```python
  with open(audio_path, "rb") as f:
      resp = requests.post(
          cfg.url,
          params={"model": cfg.model, "return_label": False},
          files={"audiofile": f},
          timeout=cfg.timeout,
      )
  resp.raise_for_status()
  data = resp.json()
  gender = data["gender"].lower()   # "MALE" → "male"
  probs = data.get("probs", [0.5, 0.5])
  confidence = max(probs)
  return {"gender": gender, "gender_confidence": confidence}
  ```

**Task 1b — Implement EnrichLabelsStage**
- **File**: `src/multitalker_asr/data/pipeline/stages/enrich_labels.py` — new
- **touches**: `[src/multitalker_asr/data/pipeline/stages/enrich_labels.py]`
- **provides**: `[EnrichLabelsStage]`
- **requires**: `[BaseStage, PipelineConfig, PipelineCheckpoint from Phase 1]`
- **depends_on**: `[Phase 1 complete]`
- Logic for `_parse_metadata_file()`:
  - Open file with `encoding="utf-8"`, iterate lines
  - Strip whitespace, skip empty lines
  - Split by `|`
  - 4 fields (nu-mien-bac):
    ```
    fields = [id_str, speaker, wav_path, text]
    gender = "female" if "nu" in speaker.lower() else None
    emotion = "neutral"
    duration = None
    ```
  - 5 fields (emotion_tongdai):
    ```
    fields = [id_str, wav_path, emotion_raw, duration_str, text]
    emotion = cfg.emotion_mapping.get(emotion_raw, "neutral")
    duration = float(duration_str)
    gender = None
    ```
  - For each line, create record dict:
    ```python
    {
        "id": Path(wav_path).stem,
        "audio_filepath": str(Path(wav_path).resolve()),
        "text": text.strip(),
        "text_itn": None,
        "emotion": emotion,
        "gender": gender,
        "language": "vi",
        "duration": duration,
    }
    ```
- `run()` calls `_parse_metadata_file(config.enrich.metadata_path, config.enrich)`
  - Note: add `metadata_path: str = ""` to `EnrichConfig` in `config.py`
  - Return parsed records (ignore input `records` — this stage creates from scratch)

### Wave 2

**Task 2a — Add metadata_path to EnrichConfig**
- **File**: `src/multitalker_asr/data/pipeline/config.py` — edit
- **touches**: `[src/multitalker_asr/data/pipeline/config.py]`
- **provides**: `[EnrichConfig.metadata_path field]`
- **requires**: `[Task 1b needs this field]`
- **depends_on**: `[Phase 1 Task 1c]`
- Add `metadata_path: str = ""` to `EnrichConfig` dataclass

**Task 2b — Update pipeline_pretranscribed.yaml**
- **File**: `configs/pipeline_pretranscribed.yaml` — edit
- **touches**: `[configs/pipeline_pretranscribed.yaml]`
- **provides**: `[metadata_path in pretranscribed config]`
- **requires**: `[Task 2a]`
- **depends_on**: `[Task 2a]`
- Add under `enrich:`:
  ```yaml
  enrich:
    input_format: pipe_delimited
    metadata_path: ""   # set at CLI: --enrich.metadata_path /path/to/metadata.txt
    emotion_mapping:
      NEU: neutral
      POS: positive
      NEG: negative
  ```

**Task 2c — Write tests**
- **File**: `tests/test_pipeline_enrichment.py` — new
- **touches**: `[tests/test_pipeline_enrichment.py]`
- **provides**: `[enrichment tests]`
- **requires**: `[GenderClassifyStage, EnrichLabelsStage]`
- **depends_on**: `[Tasks 1a, 1b]`
- Tests:
  - `test_gender_classify_lowercases()` — mock HTTP response with `"MALE"` → assert record has `"male"`
  - `test_gender_classify_skips_existing()` — record with `gender="female"` already set → assert no HTTP call
  - `test_gender_service_down_skips_stage()` — mock health_check failure → assert records returned unchanged
  - `test_gender_classify_checkpoint_skip()` — mark processed → assert no HTTP call
  - `test_enrich_nu_mien_bac_format()` — parse 4-field line, assert `gender="female"`, `emotion="neutral"`
  - `test_enrich_emotion_tongdai_format()` — parse 5-field line with `NEG`, assert `emotion="negative"`, `duration=float`
  - `test_enrich_unknown_emotion_defaults_neutral()` — unknown emotion tag → `"neutral"`

## Failure Scenarios

| When | Then | Error |
|------|------|-------|
| Gender service not running | `_check_service()` returns False → skip stage, return records unchanged | `logger.warning("Gender service unavailable at {url}, skipping gender classification")` |
| HTTP 500 from service | Log error, set `gender=None`, continue | `logger.error(f"Gender API error for {id}: {resp.status_code}")` |
| requests.Timeout | Log error, set `gender=None`, continue | `logger.error(f"Gender API timeout for {id}")` |
| Metadata file missing | `EnrichLabelsStage.run()` raises `FileNotFoundError` | Propagate — config is wrong |
| Metadata line has wrong field count | Log warning, skip line | `logger.warning(f"Unexpected field count {n} in line: {line[:80]}")` |
| WAV path in metadata doesn't exist | Include record anyway — let TranscribeStage handle missing audio | `logger.warning(f"Audio not found: {wav_path}")` |

## Rejection Criteria (DO NOT)

- DO NOT raise exception when gender service is unavailable — skip stage gracefully
- DO NOT hardcode `"female"` for all nu-mien-bac records — check speaker name field
- DO NOT use `split("|")` without stripping whitespace from each field — `[f.strip() for f in line.split("|")]`
- DO NOT make synchronous HTTP calls without timeout — always pass `timeout=cfg.timeout`
- DO NOT store raw HTTP response in record — only store parsed `gender` and `gender_confidence`
- DO NOT assume metadata line encoding is ASCII — always open with `encoding="utf-8"`

## Cross-Phase Context

**Assumes from Phase 1**: `BaseStage`, `PipelineConfig`, `PipelineCheckpoint`

**EnrichLabelsStage runs BEFORE TranscribeStage** in pretranscribed pipeline.
**GenderClassifyStage runs AFTER TranscribeStage and AlignStage** in both pipelines.

**Exports to Phase 6 (write_manifest)**:
```python
{
    ...(all prior fields),
    "gender": "male" | "female" | None,    # None if service was down
    "gender_confidence": float | None,
}
```

## Acceptance Criteria

- `uv run pytest tests/test_pipeline_enrichment.py` passes (7 tests)
- `EnrichLabelsStage` correctly parses both 4-field (nu-mien-bac) and 5-field (emotion_tongdai) formats
- `GenderClassifyStage` skips records where `gender` is already set
- If gender service is unreachable, pipeline continues (records returned without gender field modified)
- All parsed metadata fields have whitespace stripped

## Outcome Block

**What Was Planned**: HTTP-based gender classification and pipe-delimited metadata parser stages.
**Immediate Next Action**: Execute Phase 6 — implement manifest writer, pipeline orchestrator, and CLI.
**How to Measure**:
```bash
uv run pytest tests/test_pipeline_enrichment.py -v
curl -s http://localhost:8000/health_check  # verify gender service running before pipeline
python -c "
from src.multitalker_asr.data.pipeline.stages.enrich_labels import EnrichLabelsStage
print('EnrichLabelsStage imported OK')
"
```
