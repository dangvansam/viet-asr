# Phase 1 — Core Dynamic-Merge Utility + Unit Tests

**Goal:** Add a pure, GPU-free function that re-segments VAD output using a duration-adaptive silence
threshold, plus helpers, plus unit tests. No models touched.

## Data Flow

```
List[VADSegment]  (raw output from any backend, sorted or not)
      │
      ▼  dynamic_merge_segments(segments, schedule, min_dur_s, pad_s)
   sort by start
   cur = first seg; accumulated = cur.duration
   for each next seg:
       gap = next.start - cur.end
       thr = lookup_silence_s(accumulated, schedule)   # shrinks as accumulated grows
       gap <= thr ?  → MERGE (cur.end = max(cur.end,next.end); accumulated += next.duration)
                     → else CUT (emit cur; cur = next; accumulated = next.duration)
   emit final cur
      │
      ▼  merge_segments(out, min_gap_s=0.0, min_dur_s, pad_s)   # reuse for pad + min_dur filter
   List[VADSegment]  (re-cut)
```

## Code Contracts

File: `src/multitalker_asr/data/pipeline/vad_backends/base.py` (append below `assemble_frames`).

```python
def normalize_silence_schedule(
    schedule: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """[[5000,1500],[20000,800],[1e9,300]] -> sorted [(float,float)].
    A None or non-finite final limit becomes float('inf'). Empty -> raises VADBackendError."""

def lookup_silence_s(accumulated_s: float, schedule: List[Tuple[float, float]]) -> float:
    """Return silence threshold in SECONDS for current accumulated speech (seconds).
    Walk schedule (sorted ascending by limit_ms); first entry whose limit_ms >= accumulated_ms wins.
    Falls back to the last entry's silence."""

def dynamic_merge_segments(
    segments: List["VADSegment"],
    silence_schedule: List[Tuple[float, float]],
    min_dur_s: float = 0.0,
    pad_s: float = 0.0,
) -> List["VADSegment"]:
    """Re-cut/merge segments with a duration-adaptive silence threshold. Pure function.
    Empty input -> []. Schedule is normalized internally (callers may pass raw lists)."""
```

Add to imports at top of `base.py`: `from typing import Any, Dict, List, Tuple` (currently no `Tuple`).

## Tasks

### Task 1.1 — Add the three functions to `base.py`
- **File**: `src/multitalker_asr/data/pipeline/vad_backends/base.py` — edit
- **touches**: [src/multitalker_asr/data/pipeline/vad_backends/base.py]
- **provides**: [dynamic_merge_segments, lookup_silence_s, normalize_silence_schedule]
- **requires**: [VADSegment, merge_segments (already in file)]
- **Logic**:
  - `normalize_silence_schedule`: map each `(limit, silence)`; if `limit is None` or `not math.isfinite(float(limit))` → `float('inf')`; cast both to float; `sorted(..., key=lambda x: x[0])`; raise `VADBackendError` if empty. `import math` at top if absent.
  - `lookup_silence_s`: `acc_ms = accumulated_s * 1000.0`; loop normalized schedule, `if acc_ms <= limit_ms: return silence_ms / 1000.0`; after loop `return schedule[-1][1] / 1000.0`.
  - `dynamic_merge_segments`: early-return `[]` on empty. `sched = normalize_silence_schedule(silence_schedule)`. `ordered = sorted(segments, key=lambda s: s.start)`. Init `cur = VADSegment(ordered[0].start, ordered[0].end)`, `accumulated = cur.duration`. Loop rest: compute `gap`, compare to `lookup_silence_s(accumulated, sched)`; merge or cut as in Data Flow. Append final `cur`. Return `merge_segments(out, min_gap_s=0.0, min_dur_s=min_dur_s, pad_s=pad_s)`.
- **Edge cases**: single segment → returns it (after pad/min_dur filter); overlapping inputs (gap<0) → always merge (gap <= thr since thr>=0); `pad_s` applied via reused `merge_segments`.

### Task 1.2 — Export from package init
- **File**: `src/multitalker_asr/data/pipeline/vad_backends/__init__.py` — edit
- **touches**: [src/multitalker_asr/data/pipeline/vad_backends/__init__.py]
- **provides**: [public exports of the 3 helpers]
- **requires**: [Task 1.1]
- **Logic**: add `dynamic_merge_segments, lookup_silence_s, normalize_silence_schedule` to the
  `from .base import (...)` block and to `__all__`.

### Task 1.3 (TEST) — Unit tests for the core
- **File**: `tests/test_dynamic_vad.py` — new
- **touches**: [tests/test_dynamic_vad.py]
- **requires**: [Task 1.1, Task 1.2]
- **Logic** (pytest, no models, import `VADSegment` + the 3 helpers):
  - `test_normalize_inf_tail`: `[[5000,1500],[1e9,300]]` → last limit is `inf`; `[[5000,1500],[None,300]]` → last limit `inf`.
  - `test_lookup_shrinks`: schedule `[(5000,1500),(1e9,300)]` → `lookup_silence_s(2.0,·)==1.5`, `lookup_silence_s(60.0,·)==0.3`.
  - `test_short_gap_merges_when_low_accumulated`: segs `[(0,1),(1.4,2.4)]` (0.4s gap), schedule `[[5000,1500],[1e9,300]]` → 1 merged segment `(0,2.4)` (gap 0.4 <= 1.5 at acc 1s).
  - `test_same_gap_cuts_when_high_accumulated`: segs spanning >5s of accumulated speech then a 0.4s gap → 2 segments (0.4 > 0.3 once acc>5s). Construct e.g. `[(0,6),(6.4,7.4)]`: acc after first = 6s → thr 0.3 → 0.4>0.3 → cut → 2 segments.
  - `test_empty_returns_empty`: `[]` → `[]`.
  - `test_min_dur_filter`: a 0.05s lone segment with `min_dur_s=0.1` → dropped.

## Failure Scenarios

| When | Then | Error/Behavior |
|------|------|----------------|
| `silence_schedule` empty list | `normalize_silence_schedule` raises | `VADBackendError("silence_schedule is empty")` |
| `limit` is `None` / `inf` / huge | treated as `inf` tail | no crash; last bucket catches all |
| `segments` empty | return `[]` immediately | no IndexError |
| negative gap (overlap) | `gap <= thr` true (thr≥0) | merge — never produces overlapping output |
| `accumulated_s` exceeds all limits | fall through to `schedule[-1]` | smallest silence used |

## Rejection Criteria (DO NOT)
- DO NOT count the silence `gap` as speech when accumulating — only add `next.duration`.
- DO NOT mutate the caller's input list order assumption — always `sorted(...)` a copy.
- DO NOT reimplement padding / min-duration filtering — reuse existing `merge_segments`.
- DO NOT use `float('inf')` literally in any YAML or test list — use `1e9` / `None` and let normalize handle it.
- DO NOT add numpy/torch imports — this module stays pure-Python + stdlib `math`.

## Cross-Phase Context
- **Assumes from prior phases**: none (first phase).
- **Exports for Phase 2**: `dynamic_merge_segments`, `normalize_silence_schedule`, `lookup_silence_s`
  importable from `vad_backends.base` and re-exported by `vad_backends/__init__.py`.

## Acceptance Criteria
- `uv run pytest tests/test_dynamic_vad.py -q` → all pass.
- `python -c "from multitalker_asr.data.pipeline.vad_backends import dynamic_merge_segments"` → no error.
- No change to existing backend behavior (only additions to `base.py` and `__init__.py`).

## Traceability Matrix
No BA Requirements Document exists — matrix omitted (per plan HARD-GATE skip rule).

## Outcome Block
- **What Was Planned**: Pure dynamic-merge core + helpers + unit tests.
- **Immediate Next Action**: Append the three functions to `base.py`.
- **How to Measure**: `uv run pytest tests/test_dynamic_vad.py -q`
