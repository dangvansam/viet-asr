# Master Plan: Multi-Task Vietnamese ASR (Option C — Modular Hybrid)

## Goal
All-in-one streaming multi-talker Vietnamese ASR: shared encoder (pretrained or from-scratch) + SenseVoice-style prompt tokens for paralinguistic classification + RNNT decoder with EOU for streaming + Sortformer diarization + Fun-ASR-MLT-Nano as ITN/PnC post-processor.

## Architecture (Hybrid Strategy 2+3)
```
Audio (16kHz)
  │
  ├── Sortformer Module ──→ speaker activity [T×N]
  │        └──→ speaker kernels
  │
  [prompt: lang, emotion, gender, age, voice_state, textnorm] + [speech_frames]
    │
    → Shared Encoder (FastConformer or FunASR encoder, pretrained/scratch)
        ← speaker kernels injected
        │
        ├── RNNT Decoder → streaming ASR + <EOU> [per-speaker]
        ├── CE Loss on prompt positions → paralinguistic labels [per-speaker]
        │     (emotion, gender, age, voice_state, language)
        │
        └── Post-proc: Fun-ASR-MLT-Nano LLM → ITN + PnC refinement (offline)
```

Two encoder init modes:
- **Pretrained**: Load weights from Fun-ASR-MLT-Nano encoder or NeMo Parakeet
- **From scratch**: Train encoder with custom config + tokenizer

## Phases

| # | Phase | Status | Key Files | Est. LOC |
|---|-------|--------|-----------|----------|
| 1 | Foundation: Configs + Task Tokens + Prompt Embedding | ✅ | configs/multitask.py, models/prompt_embedding.py | ~200 |
| 2 | Multi-Task Model: Encoder Wrapper + Prompt + Dual Loss | ✅ | models/multitask_model.py, training/losses/ | ~400 |
| 3 | Data Pipeline: Extended Dataset + Collator + Manifest | ✅ | data/datasets/multitask.py, data/collators/multitask.py | ~250 |
| 4 | Curriculum Training: 3-Phase Trainer + Dynamic Loss | ✅ | training/curriculum_trainer.py, scripts/train_multitask.py | ~300 |
| 5 | EOU + Fun-ASR-MLT-Nano ITN/PnC Post-Processor | ✅ | models/vocab_extension.py, inference/post_processor.py | ~250 |
| 6 | Streaming Multi-Task Inference + Evaluation | ✅ | inference/multitask.py, eval/multitask_evaluator.py | ~300 |

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Paralinguistic | SenseVoice prompt-token CE loss | Proven, lightweight, streaming-compatible |
| ITN + PnC | Fun-ASR-MLT-Nano LLM post-processor | Already tested & working for Vietnamese |
| EOU | `<EOU>` token in RNNT vocabulary | NVIDIA parakeet_realtime_eou pattern, streaming |
| Encoder init | Dual mode: pretrained OR from scratch | User needs both for experimentation |
| Streaming | RNNT decoder (not LLM) | LLM is autoregressive = can't stream |
| Loss weighting | Dynamic uncertainty weighting | Prevents dominant task interference |
| Curriculum | 3-phase: ASR → multi-talker → paralinguistic | Research blueprint |
| Backbone | FastConformer now, Zipformer later | Backbone not priority |

## Dependencies & Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| FunASR encoder weight extraction | Medium | Inspect state_dict keys, map to NeMo format |
| Two-decoder inference cost (RNNT + LLM) | Medium | LLM is offline post-proc only, not real-time |
| Vietnamese annotated data scarcity | High | Pseudo-label with LLM for emotion/gender/age |
| RNNT + prompt CE gradient conflict | Medium | Dynamic uncertainty weighting + curriculum |

## References
- SenseVoice prompt: `tmp/FunASR/funasr/models/sense_voice/model.py:642-774`
- Fun-ASR-Nano LLM: `tmp/FunASR/funasr/models/fun_asr_nano/model.py`
- Fun-ASR-MLT-Nano test: `/home/samdv/test_asr_punctuation.py`
- NVIDIA EOU: `nvidia/parakeet_realtime_eou_120m-v1`
- NVIDIA Multitalker: `nvidia/multitalker-parakeet-streaming-0.6b-v1`
