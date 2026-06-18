# Phase 3 — Benchmark: Dynamic On vs Off Across Backends

**Goal:** Extend `scripts/benchmark_vad_providers.py` to compare each backend with dynamic VAD
**on vs off** for accuracy (IoU + segment-duration) and speed (RTF), then run it on real clips.

## Data Flow

```
--mode dynamic  → dynamic_pairs(hf_token, schedule)
      for each base in [silero, pyannote_seg, ten, fsmn]:
          ProviderSpec("<base>",      "<base>",  base_kwargs)          # dynamic OFF
          ProviderSpec("<base>+dyn",  "dynamic", {provider:<base>,...}) # dynamic ON
      │
      ▼  VADBenchmark.run(specs)   (existing: times load/detect, RTF, ratio, segs, frames)
      ▼  print_report(results, duration)
   existing: per-provider table + pairwise frame-IoU + disagreement-vs-majority
   NEW: avg_seg_s / median_seg_s columns
   NEW (optional): frame F1/IoU vs --ground-truth labels
```

## Code Contracts

File: `scripts/benchmark_vad_providers.py` — edit.

```python
@dataclass
class ProviderResult:        # ADD two fields
    ...
    avg_seg_s: float = 0.0
    median_seg_s: float = 0.0

def dynamic_pairs(hf_token: str, schedule=None) -> List[ProviderSpec]: ...
# main: add --mode {providers,dynamic} (default providers), --schedule (JSON), --ground-truth (path)
```

## Tasks

### Task 3.1 — Segment-duration metrics in `ProviderResult` + `_run_one`
- **File**: `scripts/benchmark_vad_providers.py` — edit
- **touches**: [scripts/benchmark_vad_providers.py]
- **provides**: [avg_seg_s, median_seg_s in results]
- **requires**: []
- **Logic**: add `avg_seg_s`, `median_seg_s` fields to `ProviderResult` (default 0.0). In `_run_one`,
  after computing `speech_seconds`: `durs = [s.duration for s in result.segments]`;
  `avg = float(np.mean(durs)) if durs else 0.0`; `med = float(np.median(durs)) if durs else 0.0`;
  pass into the returned `ProviderResult`.

### Task 3.2 — `dynamic_pairs` spec generator + CLI flags
- **File**: `scripts/benchmark_vad_providers.py` — edit
- **touches**: [scripts/benchmark_vad_providers.py]
- **provides**: [dynamic_pairs, --mode, --schedule, --ground-truth]
- **requires**: [Phase 2 "dynamic" backend]
- **Logic**:
  - `dynamic_pairs(hf_token, schedule=None)`: bases =
    `[("silero",{}), ("pyannote_seg",{"hf_token":hf_token or None}), ("ten",{}), ("fsmn",{})]`.
    For each, append OFF spec `ProviderSpec(name, name, kw)` and ON spec
    `ProviderSpec(f"{name}+dyn","dynamic", {"provider":name,"provider_kwargs":kw, **({"silence_schedule":schedule} if schedule else {})})`.
  - In `__main__`: add `--mode` (choices `providers`,`dynamic`; default `providers`),
    `--schedule` (str JSON, parsed via `json.loads` → list), `--ground-truth` (str path, default "").
    If `args.mode=="dynamic"`: `providers = dynamic_pairs(os.environ.get("HF_TOKEN",""), schedule)`.
    Keep existing `--providers` label subset filtering working for both modes.

### Task 3.3 — Report new columns + optional ground-truth F1
- **File**: `scripts/benchmark_vad_providers.py` — edit
- **touches**: [scripts/benchmark_vad_providers.py]
- **provides**: [seg-duration columns, GT F1/IoU rows]
- **requires**: [Task 3.1, 3.2]
- **Logic**: extend `print_report` header + rows with `AVG_SEG(s)` and `MED_SEG(s)`
  (`f"{r.avg_seg_s:.2f}"`). If a ground-truth frame vector is provided, add a section printing per
  provider `frame_iou(r.frames, gt)` and F1 (`2*inter/(a.sum()+b.sum())`). Loading GT: small helper
  `load_ground_truth(path, n_frames, frame_hop_s)` parsing RTTM or JSON `[{"start","end"}]` →
  rasterized int8 vector (reuse `VADBenchmark._rasterize` logic). If `--ground-truth` empty, skip.

### Task 3.4 (RUN) — Execute comparison on crawl clips
- **File**: (no file) — execution + capture
- **requires**: [3.1, 3.2, 3.3]
- **Logic**: pick 2 clips — one short single-speaker, one long multi-speaker (from crawl data or
  `demo_16k.wav`). Run on **GPU1 only** (`CUDA_VISIBLE_DEVICES=1`, per `.rune` HARD constraint) where
  a backend uses GPU:
  `CUDA_VISIBLE_DEVICES=1 uv run python scripts/benchmark_vad_providers.py --audio <clip> --mode dynamic`.
  Capture the table; note per backend: RTF on/off delta (expect ~0), segment-count and avg-duration
  shift, IoU-vs-majority change. Summarize findings back to the user.

## Failure Scenarios

| When | Then | Behavior |
|------|------|----------|
| a backend missing (ten/pyannote pkg) | existing `_run_one` try/except | row marked FAIL; others continue |
| `--schedule` malformed JSON | `json.loads` raises at startup | clear argparse-level error before any model load |
| `--ground-truth` file absent | guard: skip GT section | benchmark still prints relative metrics |
| fsmn provider already dynamic-by-default | OFF spec uses bare `fsmn` (native dynamic on) | note in summary: fsmn "OFF" ≠ truly off; pass `dynamic_silence=False` via provider_kwargs for a true-off baseline |

## Rejection Criteria (DO NOT)
- DO NOT run GPU backends on GPU0 — `CUDA_VISIBLE_DEVICES=1` is mandatory (`.rune` constraint).
- DO NOT add a new benchmark script — extend the existing one.
- DO NOT break the default `--mode providers` path — it must behave exactly as before.
- DO NOT claim accuracy gains without a reference — without `--ground-truth`, report deltas/agreement, not "better".

## Cross-Phase Context
- **Assumes from Phase 2**: `dynamic` backend builds with `{provider, provider_kwargs, silence_schedule}`.
- **Exports**: comparison table (artifact for the user / `.rune` notes); no downstream phase.

## Acceptance Criteria
- `uv run python scripts/benchmark_vad_providers.py --audio demo_16k.wav --mode providers` → unchanged output (regression).
- `--mode dynamic` → prints paired rows `<backend>` and `<backend>+dyn` with `AVG_SEG(s)`/`MED_SEG(s)` columns.
- Run on 2 clips completes; summary states per-backend on/off speed delta and segmentation change.

## Traceability Matrix
No BA Requirements Document — omitted.

## Outcome Block
- **What Was Planned**: Benchmark extension (seg-duration + GT F1) and a real on/off comparison run.
- **Immediate Next Action**: Add `avg_seg_s`/`median_seg_s` to `ProviderResult` and `_run_one`.
- **How to Measure**: `CUDA_VISIBLE_DEVICES=1 uv run python scripts/benchmark_vad_providers.py --audio <clip> --mode dynamic`
