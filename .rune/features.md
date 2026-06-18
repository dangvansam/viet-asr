# Feature Map: multitalker-asr

| Feature | Status | Key Files | Dependencies |
|---------|--------|-----------|--------------|
| Multi-talker ASR (Conformer+RNNT) | Complete | models/multitalker.py, training/trainer.py | NeMo |
| Sortformer Diarization (4-spk) | Complete | eval/evaluator.py, models/multitalker.py | NeMo Sortformer |
| On-the-fly Synthesis | Complete | data/synthesizers/, data/datasets/streaming.py | Lhotse |
| Streaming Inference | Complete | inference/streaming.py | NeMo CacheAwareStreamingAudioBuffer |
| Offline Inference | Complete | inference/offline.py | NeMo |
| Diarization Evaluation | Complete | eval/evaluator.py, eval/metrics/ | pyannote |
| Speaker/Gender/Emotion/Age Heads (MLP) | Complete | models/heads/ | PyTorch |
| **Multi-Task Paralinguistic (SenseVoice-style)** | **Planned** | models/multitask_model.py, models/prompt_embedding.py | Phase 1-2 |
| **Extended Data Pipeline** | **Planned** | data/datasets/multitask.py, data/collators/multitask.py | Phase 3 |
| **Data Processing Pipeline** | **Planned** | data/pipeline/pipeline.py, data/pipeline/stages/, scripts/prepare_data.py | Standalone |
| **Dynamic VAD (backend-agnostic)** | **Planned** | data/pipeline/vad_backends/dynamic.py, vad_backends/base.py, scripts/benchmark_vad_providers.py | Data Processing Pipeline (VAD layer) |
| **Curriculum Training** | **Planned** | training/curriculum_trainer.py | Phase 4 |
| **EOU Detection** | **Planned** | models/vocab_extension.py | Phase 5 |
| **ITN/PnC Post-Processing** | **Planned** | inference/post_processor.py | Phase 5, FunASR |
| **Multi-Task Inference + Eval** | **Planned** | inference/multitask.py, eval/multitask_evaluator.py | Phase 6 |

## Dependency Graph
```
Multi-talker ASR ← Sortformer Diarization
       ↑
Multi-Task Paralinguistic (NEW) ← Prompt Embedding + Task Tokens
       ↑
Curriculum Training (NEW) ← Extended Data Pipeline
       ↑
EOU + ITN/PnC (NEW) ← Vocab Extension + FunASR Post-Processor
       ↑
Multi-Task Inference (NEW) ← All above
```

## Known Gaps
- Zipformer backbone (deferred — backbone not priority)
- Scalable >4 speaker diarization (deferred)
- Vietnamese-specific evaluation datasets
- Paralinguistic annotated Vietnamese data — **being addressed by Data Processing Pipeline**
