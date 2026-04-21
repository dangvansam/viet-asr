# Vietnamese Multi-Task Multi-Talker ASR

An all-in-one streaming Vietnamese ASR system that handles **overlapping multi-speaker speech recognition** with **paralinguistic classification** (emotion, gender, age, voice state). Built on NVIDIA NeMo with Conformer encoder, RNNT decoder, Sortformer diarization, and SenseVoice-style prompt-based multi-task learning.

## Features

- **Multi-Talker ASR**: Transcribe overlapping speech from multiple speakers simultaneously via speaker kernel injection
- **Paralinguistic Classification**: Per-speaker emotion, gender, age, and voice state detection using SenseVoice-style prompt tokens
- **Streaming Inference**: Real-time ASR with cache-aware Conformer + End-of-Utterance (`<EOU>`) detection
- **ITN/PnC Post-Processing**: Inverse text normalization and punctuation via Fun-ASR-MLT-Nano (offline refinement)
- **3-Phase Curriculum Training**: ASR → multi-talker → paralinguistic, with dynamic loss weighting
- **On-the-Fly Synthesis**: Generate overlapping multi-speaker training data dynamically during training
- **Dual Encoder Init**: Load pretrained weights from NeMo Parakeet or FunASR, or train from scratch

---

## Architecture

```
Audio (16kHz)
  │
  ├── Sortformer Module ──→ speaker activity [T×N]
  │        └──→ speaker kernels
  │
  [prompt: lang, emotion, gender, age, voice_state, textnorm] + [speech_frames]
    │
    → FastConformer Encoder (shared) ← speaker kernels injected
        │
        ├── RNNT Decoder → streaming ASR + <EOU> [per-speaker]
        ├── CE Loss on prompt positions → paralinguistic labels [per-speaker]
        │     Position 0: Language (vi, en, zh, auto)
        │     Position 1: Emotion (happy, sad, angry, neutral, fear, disgust, surprise)
        │     Position 2: Gender (male, female)
        │     Position 3: Age (child, young, middle_age, old)
        │     Position 4: Voice state (sober, drunk)
        │     Position 5: Text normalization (with_itn, without_itn)
        │
        └── Post-proc: Fun-ASR-MLT-Nano → ITN + PnC refinement (offline)
```

### Loss Function

```
L_total = λ₁·L_RNNT + λ₂·L_SortLoss + λ₃·L_CE_prompt
```

Phase 3 uses dynamic uncertainty weighting (Kendall et al. 2018) to auto-balance task magnitudes.

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA GPU (training) / CPU (inference)
- `ffmpeg` (`sudo apt install ffmpeg`)

### Install Dependencies

```bash
uv sync
```

### Download Base Models

```bash
mkdir -p models

# Parakeet 0.6B multi-talker ASR
wget -c https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1/resolve/main/multitalker-parakeet-streaming-0.6b-v1.nemo -P models/

# Sortformer v2.1 diarization
wget -c https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1/resolve/main/diar_streaming_sortformer_4spk-v2.1.nemo -P models/
```

---

## Quick Start

### Data Preparation

```bash
# Step 1: Generate single-speaker manifests from TTS data
uv run scripts/prepare_all_tts_data.py \
    --data_root /path/to/tts_datasets \
    --output_dir data \
    --val_split 0.05

# Step 2: Generate speaker metadata template (review & edit gender/age)
uv run scripts/prepare_multitask_data.py \
    --input_manifest data/train_single_speaker.json \
    --output_manifest data/train_multitask.json \
    --generate_template \
    --speaker_metadata data/speaker_metadata.json

# Step 3: Annotate manifests with multi-task labels
uv run scripts/prepare_multitask_data.py \
    --input_manifest data/train_single_speaker.json \
    --output_manifest data/train_multitask.json \
    --speaker_metadata data/speaker_metadata.json \
    --val_manifest data/val_single_speaker.json \
    --val_output data/val_multitask.json
```

The extended manifest format:
```json
{"audio_filepath": "/path/audio.wav", "text": "xin chào", "duration": 2.5,
 "label": "nu-mien-bac", "emotion": "neutral", "gender": "female",
 "age": "young", "voice_state": "sober", "language": "vi"}
```

### Training (3-Phase Curriculum)

```bash
# Run all 3 phases sequentially
bash scripts/run_multitask_training.sh all

# Or run individual phases
bash scripts/run_multitask_training.sh 1   # Phase 1: Vietnamese ASR
bash scripts/run_multitask_training.sh 2   # Phase 2: Multi-talker
bash scripts/run_multitask_training.sh 3   # Phase 3: Paralinguistic
```

| Phase | Data | Frozen Layers | Loss | LR |
|-------|------|---------------|------|----|
| 1 - ASR | Single-speaker | 18 | RNNT | 1e-4 |
| 2 - Multi-talker | Synthetic overlap | 12 | RNNT + Sort | 1e-5 |
| 3 - Paralinguistic | Annotated multi-task | 0 | RNNT + Sort + CE | 5e-6 |

### Inference

```bash
# Basic multi-task inference
uv run scripts/infer_multitask.py \
    --audio_path test.wav \
    --model_path checkpoints/phase3_paralinguistic/last.ckpt

# With Fun-ASR-MLT-Nano ITN/PnC post-processing
uv run scripts/infer_multitask.py \
    --audio_path test.wav \
    --model_path checkpoints/phase3_paralinguistic/last.ckpt \
    --use_post_processor \
    --output_json result.json
```

Output:
```
[speaker_0] Xin chào, bạn khỏe không? (emotion=neutral, gender=female, age=young, lang=vi) [EOU]
```

### Standard Training (Single-task ASR only)

```bash
uv run scripts/train.py \
    --mode finetune \
    --model_path models/multitalker-parakeet-streaming-0.6b-v1.nemo \
    --train_manifest data/train_single_speaker.json \
    --val_manifest data/val_single_speaker.json \
    --use_on_the_fly_synthesis \
    --max_speakers 4
```

---

## Project Structure

```
src/multitalker_asr/
├── configs/
│   ├── model.py              # ASR/diarization model paths
│   ├── training.py           # Training hyperparameters
│   ├── multitask.py          # Multi-task config (tasks, prompt dims, loss weights)
│   └── ...
├── models/
│   ├── multitalker.py        # Base ASR model wrapper (NeMo EncDecMultiTalkerRNNTBPE)
│   ├── multitask_model.py    # Multi-task model (prompt prepending + dual loss)
│   ├── prompt_embedding.py   # TaskTokenRegistry + PromptEmbedding (SenseVoice-style)
│   ├── vocab_extension.py    # RNNT vocabulary extension (<EOU>)
│   └── heads/                # Speaker/gender/emotion/age MLP heads
├── training/
│   ├── trainer.py            # Base MultitalkerTrainer (PyTorch Lightning)
│   ├── curriculum_trainer.py # 3-phase curriculum training orchestrator
│   └── losses/
│       ├── multi_task_loss.py     # Weighted RNNT + prompt CE loss
│       └── dynamic_weighting.py   # Kendall uncertainty-based loss weighting
├── data/
│   ├── datasets/
│   │   ├── streaming.py      # On-the-fly multi-speaker synthesis
│   │   └── multitask.py      # Extended dataset with paralinguistic labels
│   ├── collators/
│   │   ├── multitalker.py    # Audio/text/mask padding
│   │   └── multitask.py      # + task_labels batching
│   ├── mixers/               # Audio mixing (overlapping speech)
│   └── synthesizers/         # Batch multi-speaker synthesis
├── inference/
│   ├── multitask.py          # SpeakerResult + MultitaskInferenceEngine
│   ├── post_processor.py     # Fun-ASR-MLT-Nano ITN/PnC wrapper
│   ├── streaming.py          # Cache-aware streaming inference
│   └── offline.py            # Offline transcription
├── eval/
│   ├── multitask_evaluator.py # WER/CER + per-task accuracy metrics
│   └── evaluator.py          # Diarization evaluation (DER)
└── utils/                    # Checkpoint, device, audio, text utilities

scripts/
├── train.py                  # Standard training CLI
├── train_multitask.py        # Multi-task curriculum training CLI
├── infer.py                  # Standard inference CLI
├── infer_multitask.py        # Multi-task inference CLI
├── prepare_all_tts_data.py   # TTS data → NeMo manifests
├── prepare_multitask_data.py # Annotate manifests with task labels
├── run_full_training.sh      # Standard training pipeline
└── run_multitask_training.sh # Multi-task curriculum pipeline
```

---

## Multi-Task Labels

| Task | Classes | Prompt Position |
|------|---------|----------------|
| Language | vi, en, zh, auto | 0 |
| Emotion | happy, sad, angry, neutral, fear, disgust, surprise | 1 |
| Gender | male, female | 2 |
| Age | child, young, middle_age, old | 3 |
| Voice State | sober, drunk | 4 |
| Text Norm | with_itn, without_itn | 5 |

---

## Tech Stack

- **Framework**: PyTorch + NVIDIA NeMo + PyTorch Lightning
- **ASR**: Conformer encoder + RNNT decoder (Parakeet 0.6B)
- **Diarization**: Streaming Sortformer (4-speaker)
- **Multi-task**: SenseVoice-style prompt token embedding + CE loss
- **ITN/PnC**: Fun-ASR-MLT-Nano (LLM-based, offline)
- **Package Manager**: uv
- **Tokenizer**: SentencePiece BPE

## References

- [NVIDIA Multitalker Parakeet](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)
- [NVIDIA Streaming Sortformer](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)
- [FunAudioLLM SenseVoice](https://github.com/FunAudioLLM/SenseVoice)
- [FunAudioLLM Fun-ASR-MLT-Nano](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512)
- [NVIDIA Parakeet EOU](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1)

## Testing

```bash
uv run python -m pytest tests/ -v
```

82 tests covering all multi-task components (configs, prompt embedding, model, data pipeline, curriculum training, loss weighting, inference, evaluation).
