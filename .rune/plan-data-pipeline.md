# Plan: Data Processing Pipeline (data-pipeline)

## Goal
Build a crash-resumable, stage-based data pipeline in `src/multitalker_asr/data/pipeline/`
that converts raw audio/video and pre-transcribed datasets into NeMo JSONL manifests
with 6 multitask labels: `text`, `emotion`, `gender`, `language`, `textnorm`, `alignment`.

## Phases

| # | Phase | Key Output | Status |
|---|-------|------------|--------|
| 1 | Foundation: Config + Checkpoint + BaseStage | `pipeline/config.py`, `pipeline/checkpoint.py`, `pipeline/base_stage.py` | ✅ Done |
| 2 | Preprocessing Stages: extract_audio + vad_diarize | `stages/extract_audio.py`, `stages/vad_diarize.py` | ✅ Done |
| 3 | Transcription Stage: FunASR MLT-Nano | `stages/transcribe.py` (audio→text+ITN+emotion) | ✅ Done |
| 4 | Alignment Stage: Qwen3ForcedAligner | `stages/align.py` (text+audio→word timestamps) | ✅ Done |
| 5 | Enrichment Stages: gender + label mapping | `stages/gender_classify.py`, `stages/enrich_labels.py` | ✅ Done |
| 6 | Manifest + Orchestrator: write + CLI | `stages/write_manifest.py`, `pipeline.py`, `scripts/run_pipeline.py`, YAML configs | ✅ Done |

## Architecture

```
scripts/prepare_data.py --config pipeline_raw.yaml --input_dir X --output_dir Y
    └── pipeline.py (DataPipeline)
          ├── checkpoint.py (PipelineCheckpoint — atomic JSON state)
          ├── config.py (PipelineConfig dataclass)
          └── stages/ (each stage: run(records) → records + checkpoint.mark_processed)
                extract_audio   → ffmpeg video→wav
                vad_diarize     → pyannote-onnx → segments
                transcribe      → FunASR MLT-Nano → text+ITN+emotion+language
                align           → Qwen3ForcedAligner-0.6B → word timestamps
                gender_classify → HTTP POST /predict → gender label
                enrich_labels   → parse pipe-delimited metadata, map labels
                write_manifest  → NeMo JSONL with all 6 task labels
```

## Two Pipeline Configs

| Config | Input | Stages Run |
|--------|-------|-----------|
| `pipeline_raw.yaml` | Long audio/video (TikTok/YT) | extract_audio → vad_diarize → transcribe → align → gender_classify → write_manifest |
| `pipeline_pretranscribed.yaml` | Pipe-delimited metadata (nu-mien-bac, emotion_tongdai) | enrich_labels → transcribe → gender_classify → write_manifest |

## Key Decisions

- **Checkpoint**: Adapted from `/home/samdv/data-processing-pipeline/pipeline/checkpoint.py` — atomic JSON, per-stage file tracking
- **Aligner**: `Qwen3ForcedAligner.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B")` — adapted from reference `alignment.py`
- **VAD**: pyannote-onnx (already at `/home/samdv/pyannote-onnx`, git submodule in reference pipeline)
- **Transcription**: FunASR `AutoModel("FunAudioLLM/Fun-ASR-MLT-Nano-2512")` — returns text + ITN + emotion + language natively
- **Gender**: HTTP POST `http://localhost:8000/predict` (gender-classification-service must be running)
- **GPU memory**: freed with `gc.collect() + torch.cuda.empty_cache()` between heavy stages (Phase 3, 4)

## Constraints
1. Crash at stage N → resume from stage N (not stage 1)
2. GPU memory freed between FunASR and Qwen3 stages
3. Both pipelines share stage implementations (different YAML configs only)
4. `pipeline_pretranscribed.yaml`: supports `itn_only` mode (text normalization without re-transcribing audio)
5. Final manifest compatible with existing NeMo training pipeline

## Dependencies / Risks
- pyannote-onnx HF token required for first download
- Gender service must be running at localhost:8000 during pipeline execution
- Qwen3-ForcedAligner-0.6B ~1.2GB download on first run
- FunASR MLT-Nano model ~600MB download on first run

## Addresses Gap
Fills "Paralinguistic annotated Vietnamese data (need pseudo-labeling)" from `.rune/features.md`
