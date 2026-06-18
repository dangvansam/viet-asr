# Plan: Backend-Agnostic Dynamic VAD (dynamic-vad)

## Goal
Bring FunASR's **Dynamic VAD** (adaptive silence-cut threshold that shrinks as speech accumulates)
to **every** VAD backend — silero, pyannote_seg, ten, fsmn, consensus — as a composable wrapper,
controllable from `backend_kwargs`. Then benchmark accuracy + speed with dynamic on vs off per backend.

## Background
FunASR implements dynamic silence internally only in its FSMN state machine
(`tmp/FunASR/.../fsmn_vad_streaming/model.py:942-999`, on-by-default). Every backend returns the
same `VADResult{segments:[VADSegment(start,end)], speech_ratio}`, so the silence schedule can be
applied as a backend-agnostic **post-process on output segments**. Mirrors the existing
`ConsensusVADBackend` meta-backend pattern.

## Phases

| # | Phase | Key Output | Status |
|---|-------|------------|--------|
| 1 | Core utility + unit tests | `vad_backends/base.py` (`dynamic_merge_segments`, `lookup_silence_s`, `normalize_silence_schedule`) + `tests/test_dynamic_vad.py` | ✅ Done |
| 2 | Wrapper backend + config | `vad_backends/dynamic.py` (`DynamicVADBackend`), register in `__init__.py`, FSMN passthrough, `configs/pipeline_crawl_vad_dynamic.yaml` | ✅ Done |
| 3 | Benchmark + comparison run | extend `scripts/benchmark_vad_providers.py` (`--mode dynamic`, seg-duration + GT F1), ran on crawl clips | ✅ Done |

## Architecture

```
build_vad_backend("dynamic", provider="silero", silence_schedule=[...])
    └── DynamicVADBackend (vad_backends/dynamic.py)          # mirrors ConsensusVADBackend
          ├── inner = build_vad_backend(provider, **provider_kwargs)   # any registered backend
          ├── base = inner.detect(audio, sr)                # raw VADResult
          └── dynamic_merge_segments(base.segments, schedule)  # base.py — pure, GPU-free
                 walk L→R, accumulate speech since last cut;
                 gap <= lookup_silence_s(accumulated) → merge, else cut
```

## Key Decisions
- **Wrapper, not per-backend edits**: one `DynamicVADBackend` composes over any provider (incl.
  `consensus`) — matches `consensus.py` pattern, plugs into VADStage/VADDiarizeStage with no stage change.
- **Core logic is pure**: `dynamic_merge_segments` = `merge_segments` with a duration-adaptive
  `min_gap_s`. No models, fully unit-testable.
- **Schedule is YAML-friendly**: `[[5000,1500],[1e9,300]]` lists; `normalize_silence_schedule`
  treats a null/huge final limit as `inf` (YAML has no `inf`).
- **FSMN native passthrough optional** (Phase 2): FSMN can do dynamic at frame level; expose
  `dynamic_silence`/`silence_schedule`/`speech_noise_thres` kwargs → `generate()` as a secondary path.
- **Default schedule** = FunASR's `[(5000,2000),(10000,1500),(15000,1000),(30000,800),(45000,400),(inf,100)]`.

## Decision Compliance
- No BA Requirements Document exists for this feature — no locked decisions to honor.
- Respects `.rune/decisions.md`: VAD layer stays "pluggable single-select backend" — the wrapper is
  another selectable backend, not a replacement.

## Constraints
1. `VADResult` shape unchanged; `VADConfig.backend_kwargs` already generic → no schema/stage edits.
2. Additive only — `VAD_REGISTRY` grows by one key; existing configs untouched (regression-safe).
3. Every code-producing phase ships its own tests (Phase 1 unit tests; Phase 3 benchmark = exec proof).
4. Dynamic is a cheap post-process → detect-time RTF must stay ~equal to the bare provider.

## Dependencies / Risks
- Benchmark "accuracy" without ground-truth labels falls back to IoU-vs-majority + segment-duration
  deltas; optional `--ground-truth` adds frame F1/IoU when labels exist.
- pyannote_seg provider needs HF token (existing constraint, threaded via `hf_token` kwarg).

## Addresses Gap
Strengthens the VAD/segmentation layer noted in `.rune/features.md` — segment boundary quality
directly affects ASR agreement and manifest segment durations.

## Outcome Block
- **What Was Planned**: A composable dynamic-VAD wrapper over all backends + a comparison benchmark.
- **Immediate Next Action**: Implement Phase 1 — add `dynamic_merge_segments` to `base.py` and its unit test.
- **How to Measure**:

| Check | Command |
|-------|---------|
| Unit tests pass | `uv run pytest tests/test_dynamic_vad.py -q` |
| Wrapper registered | `uv run python -c "from multitalker_asr.data.pipeline.vad_backends import list_vad_backends; print('dynamic' in list_vad_backends())"` |
| Benchmark on/off | `uv run python scripts/benchmark_vad_providers.py --audio <clip> --mode dynamic` |

---
## Results (2026-06-18) — ALL backends

Benchmark on a 51.7s crawl clip, dynamic off vs on, CPU, tuned schedule `[[5000,1500],[20000,800],[1e9,300]]`:

| Backend | Segs off→on | Avg seg (s) off→on | Detect (s) off→on | IoU vs fsmn off→on |
|---------|-------------|--------------------|--------------------|--------------------|
| silero | 27 → 3 | 1.17 → 13.93 | 0.58 → 0.53 | 0.731 → 0.962 |
| ten | 33 → 4 | 0.95 → 11.41 | 0.32 → 0.33 | 0.720 → 0.889 |
| fsmn | 1 → 1 | 43.43 → 43.43 | 0.17 → 0.14 | (ref) |
| consensus (silero+ten) | 26 → 4 | 1.34 → 11.48 | 0.90 → 0.85 | — |
| pyannote_seg | 24 → 3 | 1.29 → 13.83 | 0.97 → 0.91 | 0.758 (off→on) |

**pyannote_seg now works in-process** after upgrading `pyannote.audio` 3.4.0 → **4.0.4** (4.x matches
torchaudio 2.10; 3.x referenced the removed `torchaudio.AudioMetaData`). torchaudio stayed 2.10.0+cu128;
full test suite unchanged (478 passed). torchcodec `.so` warning is harmless — the backend passes the
waveform tensor directly, no file decode.

Short clip (4.4s): every backend 1 seg, on == off (accumulation < 5s never crosses a threshold) — correct.

**Findings**
- Dynamic merging works across **all** in-process backends at ~zero detect cost (cheap post-process; RTF <0.02).
- **Convergence**: dynamic pulls silero/ten toward FSMN's coarse structure — IoU silero+dyn vs fsmn 0.731→0.962, ten 0.720→0.889; disagreement-vs-majority drops silero 0.201→0.007, ten 0.213→0.076. Backends agree far more once dynamically re-cut.
- Schedule knob controls aggressiveness (default sched → silero 2 segs; tuned → 3 segs).
- Wrapper composes over `consensus` too (26→4 segs).
- **fsmn off ≈ on**: FunASR's FSMN native dynamic is on by default, so the wrapper has nothing to merge — pass `dynamic_silence=False` in `provider_kwargs` for a true-off FSMN baseline.
- **pyannote_seg / service** are not in-process testable here: pyannote.audio needs torchaudio 2.11 which conflicts with this env's torch 2.10.0+cu128 (`AudioMetaData` error) — it runs as the HTTP `service` backend by design.

**Env to run the benchmark** (model deps live outside pyproject):
`HF_TOKEN` (from `.env`), `PYTHONPATH=tmp/FunASR` (patched funasr fork, fsmn), `LD_LIBRARY_PATH=<NDK>/lib` (ten-vad libc++), `CUDA_VISIBLE_DEVICES=1`.

### REST API results (`service` backend via `scripts/serve_vad.py`)

Each backend served behind `POST /v1/audio/vad` (LitServe), benchmarked through the `service`
client + `dynamic`-over-`service`, 51.7s clip:

| Backend (REST) | Detect(s) | RTF | Segs | Avg seg(s) |
|----------------|-----------|-----|------|------------|
| silero svc | 0.516 | 0.0100 | 27 | 1.17 |
| silero svc+dyn | 0.518 | 0.0100 | 3 | 13.93 |
| fsmn svc | 0.164 | 0.0032 | 1 | 43.43 |
| fsmn svc+dyn | 0.189 | 0.0037 | 1 | 43.43 |
| ten svc | 0.340 | 0.0066 | 32 | 0.99 |
| ten svc+dyn | 0.335 | 0.0065 | 4 | 11.08 |
| pyannote svc | 0.887 | 0.0172 | 24 | 1.29 |
| pyannote svc+dyn | 0.893 | 0.0173 | 3 | 13.83 |

pyannote REST matches in-process exactly (24→3 segs); served locally via `serve_vad.py --backend
pyannote_seg` on :9504 (no docker needed once pyannote.audio 4.x is in the venv).

- REST overhead negligible (~20–30ms over a 51.7s clip): silero svc 0.516s vs in-proc 0.494s; segment
  counts match in-process within noise.
- **`dynamic` wraps `provider=service`** → dynamic VAD on REST-served segments while keeping the
  pipeline venv model-free (the whole point of the service architecture). silero 27→3, ten 32→4.
- Ports used: silero 9501, fsmn 9502, ten 9503 (9000/9001 are MinIO). fsmn service needs
  `PYTHONPATH=tmp/FunASR`, ten needs NDK `LD_LIBRARY_PATH`.
- **pyannote_seg over REST** needs the docker service (`docker compose up vad` with its own torchaudio
  2.11 env) — not a local-venv service, due to the same in-process torchaudio conflict.

**Status: Done.** 14/14 dynamic-VAD tests pass; full suite 478 passed / 1 pre-existing streaming failure (unrelated); default benchmark mode unchanged (regression-safe). REST path verified for silero/fsmn/ten incl. dynamic-over-service.
