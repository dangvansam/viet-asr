# Phase 2 — DynamicVADBackend Wrapper + Config

**Goal:** Add a `dynamic` meta-backend that wraps any registered provider and applies the Phase 1
core to its output. Register it. Optionally expose FSMN-native kwargs. Add a config example.

## Data Flow

```
VADConfig.backend="dynamic"  +  backend_kwargs={provider, provider_kwargs, silence_schedule, ...}
      │   (stages/vad_diarize.py _load_vad_backend → build_vad_backend("dynamic", **kwargs))
      ▼
DynamicVADBackend.load(device)
      └── inner = build_vad_backend(provider, **provider_kwargs); inner.load(device)
DynamicVADBackend.detect(audio, sr)
      ├── base = inner.detect(audio, sr)                      # raw VADResult
      ├── segs = dynamic_merge_segments(base.segments, schedule, min_dur_s, pad_s)   # Phase 1
      └── return VADResult(segments=segs, speech_ratio=recomputed, backend="dynamic",
                           raw={provider, base_segments, dynamic_segments, schedule})
```

## Code Contracts

File: `src/multitalker_asr/data/pipeline/vad_backends/dynamic.py` (new). Model after `consensus.py`.

```python
class DynamicVADBackend(BaseVADBackend):
    name = "dynamic"
    DEFAULT_SILENCE_SCHEDULE = [(5000,2000),(10000,1500),(15000,1000),(30000,800),(45000,400),(1e9,100)]

    def __init__(self, provider="silero", provider_kwargs=None, silence_schedule=None,
                 min_dur_s=0.0, pad_s=0.0, hf_token=None): ...
    def load(self, device="cpu") -> None: ...        # build + load inner provider
    def detect(self, audio, sample_rate) -> VADResult: ...
    def unload(self) -> None: ...                    # delegate to inner
```

## Tasks

### Task 2.1 — Implement `DynamicVADBackend`
- **File**: `src/multitalker_asr/data/pipeline/vad_backends/dynamic.py` — new
- **touches**: [src/multitalker_asr/data/pipeline/vad_backends/dynamic.py]
- **provides**: [DynamicVADBackend]
- **requires**: [dynamic_merge_segments, normalize_silence_schedule (Phase 1); build_vad_backend]
- **Logic**:
  - `__init__`: store `self._provider_name=provider`, `self._provider_kwargs=dict(provider_kwargs or {})`,
    `self._schedule = normalize_silence_schedule(silence_schedule or self.DEFAULT_SILENCE_SCHEDULE)`,
    `self._min_dur_s`, `self._pad_s`, `self._hf_token`, `self._backend=None`, `self._loaded=False`.
  - `load`: `from . import build_vad_backend` (lazy, like consensus.py avoids circular import).
    `kwargs = dict(self._provider_kwargs)`; if `self._provider_name in ("pyannote_seg","consensus")` and
    `self._hf_token` → `kwargs.setdefault("hf_token", self._hf_token)`. Build inner, `inner.load(device)`,
    set `self._backend`, `self._loaded=True`, `logger.info(...)`.
  - `detect`: guard not loaded → `VADBackendError`. `base = self._backend.detect(audio, sample_rate)`.
    `segments = dynamic_merge_segments(base.segments, self._schedule, self._min_dur_s, self._pad_s)`.
    `total = len(audio)/sample_rate if sample_rate>0 else 0.0`; `speech = sum(s.duration for s in segments)`;
    `ratio = speech/total if total>0 else 0.0`. Return `VADResult(segments, ratio, self.name,
    raw={"provider": self._backend.name, "schedule": self._schedule,
    "base_segments": len(base.segments), "dynamic_segments": len(segments)})`.
  - `unload`: if `self._backend`: `self._backend.unload()`; reset `self._backend=None`, `self._loaded=False`.
- **Edge cases**: provider build/load failure propagates (don't silently swallow — unlike consensus which
  has N providers, here there is exactly one). `base.segments` empty → `dynamic_merge_segments` returns `[]`.

### Task 2.2 — Register `dynamic` in the registry
- **File**: `src/multitalker_asr/data/pipeline/vad_backends/__init__.py` — edit
- **touches**: [src/multitalker_asr/data/pipeline/vad_backends/__init__.py]
- **provides**: ["dynamic" registry key]
- **requires**: [Task 2.1]
- **Logic**: `from .dynamic import DynamicVADBackend`; add `"dynamic": DynamicVADBackend` to
  `VAD_REGISTRY`; add `"DynamicVADBackend"` to `__all__`.

### Task 2.3 (OPTIONAL) — FSMN native dynamic passthrough
- **File**: `src/multitalker_asr/data/pipeline/vad_backends/fsmn_vad.py` — edit
- **touches**: [src/multitalker_asr/data/pipeline/vad_backends/fsmn_vad.py]
- **provides**: [tunable FSMN dynamic kwargs]
- **requires**: []
- **Logic**: add `__init__` params `dynamic_silence=None`, `silence_schedule=None`,
  `speech_noise_thres=None`, `max_end_silence_time=None` (store on `self._...`). In `detect`, build
  `gen_kwargs` with only the non-`None` values and `self._model.generate(input=..., **gen_kwargs)`.
  Add `from typing import List, Optional, Tuple` imports. Skip this task if time-boxed — wrapper covers FSMN too.

### Task 2.4 — Config example
- **File**: `configs/pipeline_crawl_vad_dynamic.yaml` — new
- **touches**: [configs/pipeline_crawl_vad_dynamic.yaml]
- **requires**: [Task 2.2]
- **Logic**: copy `configs/pipeline_crawl_vad_consensus.yaml`; set the `vad:` block to
  `backend: dynamic`, `device: cuda:0`, `enable_trim: true`, and
  `backend_kwargs: {provider: silero, provider_kwargs: {threshold: 0.5},
  silence_schedule: [[5000,1500],[20000,800],[1000000000,300]], min_dur_s: 0.0, pad_s: 0.0}`.
  Keep all other blocks (source/asr/diarize/etc.) identical to the consensus config.

### Task 2.5 (TEST) — Wrapper integration test (cpu, silero)
- **File**: `tests/test_dynamic_vad.py` — edit (append)
- **touches**: [tests/test_dynamic_vad.py]
- **requires**: [Task 2.1, 2.2]
- **Logic**: `test_dynamic_backend_registered`: `"dynamic" in list_vad_backends()`.
  `test_dynamic_wrapper_silero` (mark `@pytest.mark.skipif` if `silero_vad`/audio unavailable):
  build `dynamic` over `silero` on a synthetic 3s array, `load("cpu")`, `detect`, assert
  `result.backend=="dynamic"` and `result.raw["provider"]=="silero"`.

## Failure Scenarios

| When | Then | Error |
|------|------|-------|
| inner provider name not in registry | `build_vad_backend` raises in `load` | `ValueError("Unknown VAD backend ...")` — propagate |
| inner provider load fails (missing pkg) | propagate from `load` | original `VADBackendError`/`ImportError` |
| `detect` before `load` | guard raises | `VADBackendError("DynamicVADBackend not loaded.")` |
| `silence_schedule` malformed in YAML | `normalize_silence_schedule` raises at `__init__` | `VADBackendError` (fail fast, before pipeline run) |

## Rejection Criteria (DO NOT)
- DO NOT swallow inner-provider load errors (consensus tolerates N-1 failures; dynamic has exactly one provider).
- DO NOT import `dynamic` at module top of `__init__.py` in a way that recreates a circular import —
  follow consensus.py's lazy `from . import build_vad_backend` inside `load`.
- DO NOT re-implement segment merging here — call `dynamic_merge_segments` from Phase 1.
- DO NOT change `VADConfig` or any pipeline stage — `backend_kwargs` already flows verbatim.
- DO NOT hardcode the schedule in `detect` — it lives on `self._schedule`, normalized once in `__init__`.

## Cross-Phase Context
- **Assumes from Phase 1**: `dynamic_merge_segments`, `normalize_silence_schedule` exist and are tested.
- **Exports for Phase 3**: `dynamic` is a buildable backend; benchmark can spawn
  `ProviderSpec(label, "dynamic", {"provider": <name>, ...})`.

## Acceptance Criteria
- `python -c "from multitalker_asr.data.pipeline.vad_backends import build_vad_backend; build_vad_backend('dynamic', provider='silero')"` → constructs without error.
- `"dynamic" in list_vad_backends()` → True.
- `uv run pytest tests/test_dynamic_vad.py -q` → all pass (silero test skipped if pkg absent).
- `configs/pipeline_crawl_vad_dynamic.yaml` parses (yaml.safe_load).

## Traceability Matrix
No BA Requirements Document — omitted.

## Outcome Block
- **What Was Planned**: `DynamicVADBackend` wrapper, registry entry, config example, integration test.
- **Immediate Next Action**: Create `vad_backends/dynamic.py` with `DynamicVADBackend`.
- **How to Measure**: `uv run pytest tests/test_dynamic_vad.py -q`
