# Architecture Decisions

| Date | Decision | Rationale | Status |
|------|----------|-----------|--------|
| 2026-06-09 | Keep `qwen3` as default forced-alignment backend for Vietnamese | Empirical probe on a real vivoice clip: Qwen3-ForcedAligner returned 13 monotonic word spans with correct text despite Vietnamese not being in its 11 official languages. `nemo_nfa`/`mms_fa` remain pluggable fallbacks. | Adopted |
| 2026-06-09 | Pluggable single-select VAD + align backend layers with cross-signal consensus gate | Pipeline cuts by (segment, speaker, timestamp) from VAD + diarization + alignment; segment kept only when signals agree within tolerance + ASR-ensemble transcript feeds alignment. | Implemented |
