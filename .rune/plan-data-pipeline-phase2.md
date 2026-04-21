# Phase 2: Preprocessing Stages — extract_audio + vad_diarize

## Goal
Implement two stages that convert raw video/audio files into speaker-segmented WAV clips
written to disk, producing a JSONL record per segment with timestamps and speaker ID.

## Data Flow
```
Input: List[Dict] with {id, audio_filepath, source_video (optional)}

Stage: ExtractAudioStage
    │  ffmpeg: video/any-format → 16kHz mono WAV
    │  Skips already-extracted (checkpoint)
    ▼
records: {id, audio_filepath (wav), source_video, duration}

Stage: VADDiarizeStage
    │  pyannote-onnx Pipeline("pyannote/speaker-diarization-3.1")
    │  Processes each long WAV → speaker turn segments
    │  Clips segment WAVs → output_dir/segments/
    │  Filters: min_duration ≤ clip_duration ≤ max_duration
    ▼
records (1 per segment): {
    id,                     # "{source_id}_{speaker}_{start:.2f}_{end:.2f}"
    audio_filepath,         # path to clipped WAV segment
    source_audio,           # original long WAV path
    start, end,             # float seconds
    speaker_id,             # "SPEAKER_00", "SPEAKER_01", etc.
    duration                # end - start
}
```

## Code Contracts

```python
# src/multitalker_asr/data/pipeline/stages/extract_audio.py
import subprocess
from pathlib import Path
from typing import List, Dict
from loguru import logger
from ..base_stage import BaseStage
from ..config import PipelineConfig
from ..checkpoint import PipelineCheckpoint

class ExtractAudioStage(BaseStage):
    name = "extract_audio"

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        """
        Convert video/audio files to 16kHz mono WAV.
        Skips files already converted (checkpoint).
        Returns records with updated audio_filepath pointing to WAV.
        """
        ...

    def _extract_one(self, src_path: str, out_path: str) -> float:
        """
        Run ffmpeg: src → 16kHz mono WAV. Returns duration in seconds.
        Raises subprocess.CalledProcessError on ffmpeg failure.
        Command: ffmpeg -i src -ar 16000 -ac 1 -y out_path
        """
        ...
```

```python
# src/multitalker_asr/data/pipeline/stages/vad_diarize.py
import torchaudio
from pathlib import Path
from typing import List, Dict, Tuple
from loguru import logger
from ..base_stage import BaseStage
from ..config import PipelineConfig, VADConfig
from ..checkpoint import PipelineCheckpoint

class VADDiarizeStage(BaseStage):
    name = "vad_diarize"

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        """
        Run speaker diarization on long WAV files.
        Clips each speaker turn to a separate WAV file.
        Filters clips by min/max duration.
        Returns one record per accepted segment.
        """
        ...

    def _load_pipeline(self, vad_config: VADConfig, device: str):
        """
        Load pyannote diarization pipeline.
        Uses pyannote.audio.Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=vad_config.hf_token
        ) if hf_token set, else loads from local cache.
        """
        ...

    def _clip_segment(
        self,
        waveform,          # torch.Tensor [1, N]
        sample_rate: int,
        start: float,
        end: float,
        out_path: str,
    ) -> None:
        """Slice waveform tensor and save to out_path as WAV."""
        ...

    def _make_segment_id(
        self, source_id: str, speaker: str, start: float, end: float
    ) -> str:
        """Return "{source_id}_{speaker}_{start:.2f}_{end:.2f}"."""
        ...
```

## Tasks

### Wave 1

**Task 1a — Implement ExtractAudioStage**
- **File**: `src/multitalker_asr/data/pipeline/stages/extract_audio.py` — new
- **touches**: `[src/multitalker_asr/data/pipeline/stages/extract_audio.py]`
- **provides**: `[ExtractAudioStage]`
- **requires**: `[BaseStage, PipelineConfig, PipelineCheckpoint from Phase 1]`
- **depends_on**: `[Phase 1 complete]`
- Logic:
  1. Call `_skip_processed(records, checkpoint)` → split into `to_process`, `done`
  2. For each record in `to_process`:
     - Determine output WAV path: `output_dir/extracted/{record_id}.wav`
     - Call `_extract_one(src, out)` with ffmpeg
     - On success: update `record["audio_filepath"] = out_path`, set `record["duration"]`
     - Call `checkpoint.mark_processed(record["id"], self.name)`
  3. Save checkpoint once per batch: call `checkpoint.save_state()` after each file
  4. Return `done + processed`
- Edge cases:
  - Input is already WAV at 16kHz: ffmpeg still runs (idempotent, `-y` overwrites)
  - Input is MP4/MKV/WebM: ffmpeg extracts audio track only
  - ffmpeg not in PATH: raise `RuntimeError("ffmpeg not found — install with: apt install ffmpeg")`

**Task 1b — Implement VADDiarizeStage**
- **File**: `src/multitalker_asr/data/pipeline/stages/vad_diarize.py` — new
- **touches**: `[src/multitalker_asr/data/pipeline/stages/vad_diarize.py]`
- **provides**: `[VADDiarizeStage]`
- **requires**: `[BaseStage, PipelineConfig, PipelineCheckpoint from Phase 1]`
- **depends_on**: `[Phase 1 complete]`
- Logic:
  1. Load pyannote pipeline once (outside record loop) — store as `self._pipeline`
  2. For each record not yet in checkpoint:
     - `diarization = self._pipeline(record["audio_filepath"])`
     - Load waveform with `torchaudio.load()`
     - For each `(turn, speaker)` in `diarization.itertracks(yield_label=True)`:
       - `dur = turn.end - turn.start`
       - Skip if `dur < config.vad.min_duration or dur > config.vad.max_duration`
       - Generate segment id with `_make_segment_id()`
       - `_clip_segment()` → save WAV to `output_dir/segments/{seg_id}.wav`
       - Append new record dict to results
     - `checkpoint.mark_processed(record["id"], self.name)`
     - `checkpoint.save_state()`
  3. Free model: `del self._pipeline; gc.collect(); torch.cuda.empty_cache()`
  4. Return all segment records
- Edge cases:
  - Audio shorter than `min_duration`: entire file produces 0 segments — log warning, skip
  - pyannote returns 0 speakers: log warning, skip file
  - HF token not set: log `logger.warning("HF_TOKEN not set — pyannote may fail for first download")`

### Wave 2 (after Wave 1)

**Task 2a — Write tests**
- **File**: `tests/test_pipeline_preprocessing.py` — new
- **touches**: `[tests/test_pipeline_preprocessing.py]`
- **provides**: `[preprocessing tests]`
- **requires**: `[ExtractAudioStage, VADDiarizeStage]`
- **depends_on**: `[Tasks 1a, 1b]`
- Tests:
  - `test_extract_audio_wav_created()` — run stage on a short test MP3, assert WAV exists at expected path
  - `test_extract_audio_checkpoint_skip()` — mark file processed, run stage again, assert ffmpeg NOT called second time
  - `test_extract_audio_ffmpeg_missing()` — mock subprocess to raise FileNotFoundError, assert RuntimeError raised
  - `test_vad_diarize_segment_ids()` — test `_make_segment_id()` formatting
  - `test_vad_diarize_duration_filter()` — mock diarization output with turns of 1s, 5s, 40s; assert only 5s segment passes with defaults (3–30s)
  - `test_vad_diarize_clip_segment()` — verify clipped WAV has correct duration

## Failure Scenarios

| When | Then | Error |
|------|------|-------|
| ffmpeg binary missing | `_extract_one` raises RuntimeError with install hint | `RuntimeError: ffmpeg not found — install with: apt install ffmpeg` |
| ffmpeg fails on corrupt file | Log error, skip record, continue | `logger.error(f"ffmpeg failed for {src}: {e}")` |
| pyannote HF token expired | pyannote raises AuthenticationError | Propagate — user must fix HF token |
| WAV is mono 8kHz (telephony) | ffmpeg resamples to 16kHz — works fine | No error |
| Segment output dir full (disk) | `torchaudio.save` raises OSError | Propagate — do not silently skip |
| All segments filtered out (too short/long) | Return empty list for this source file | `logger.warning(f"No segments kept from {audio_path}")` |

## Rejection Criteria (DO NOT)

- DO NOT load the pyannote model inside the per-record loop — load ONCE before the loop
- DO NOT use `os.system()` for ffmpeg — use `subprocess.run(..., check=True)` for error checking
- DO NOT keep the full waveform tensor in memory for all files — load per-file, del after clipping
- DO NOT write segments to the input directory — always write to `output_dir/segments/`
- DO NOT swallow ffmpeg errors with bare `except: pass` — log and skip, or raise

## Cross-Phase Context

**Assumes from Phase 1**:
- `BaseStage` with `_skip_processed()` method
- `PipelineConfig` with `config.vad.min_duration`, `config.vad.max_duration`, `config.vad.hf_token`
- `PipelineCheckpoint` with `is_processed`, `mark_processed`, `save_state`
- Record dict minimum keys: `{"id": str, "audio_filepath": str}`

**Exports to Phase 3 (transcribe)**:
Each output record contains:
```python
{
    "id": str,                  # "{source_id}_{speaker}_{start:.2f}_{end:.2f}"
    "audio_filepath": str,      # abs path to clipped 16kHz mono WAV
    "source_audio": str,        # original long WAV
    "start": float,             # segment start time in source
    "end": float,               # segment end time in source
    "speaker_id": str,          # "SPEAKER_00", "SPEAKER_01", ...
    "duration": float           # end - start
}
```

## Acceptance Criteria

- `uv run pytest tests/test_pipeline_preprocessing.py` passes (6 tests)
- Given a 30-second test WAV with 2 speakers, `VADDiarizeStage.run()` returns ≥ 1 record
- Each output segment WAV exists at `output_dir/segments/` and is between 3–30 seconds
- Re-running `ExtractAudioStage` on already-processed files does not call ffmpeg again (checkpoint skip)
- No pyannote model loaded until first call to `VADDiarizeStage.run()`

## Outcome Block

**What Was Planned**: ffmpeg audio extraction and pyannote-based VAD+diarization into speaker segments.
**Immediate Next Action**: Execute Phase 3 — implement `stages/transcribe.py` using FunASR MLT-Nano.
**How to Measure**:
```bash
uv run pytest tests/test_pipeline_preprocessing.py -v
python -c "
from src.multitalker_asr.data.pipeline.stages.extract_audio import ExtractAudioStage
print('ExtractAudioStage imported OK')
"
```
