# Vietnamese Multitalker ASR

A complete pipeline for training, fine-tuning, and deploying a Vietnamese multi-speaker ASR system based on NVIDIA's `multitalker-parakeet-streaming-0.6b-v1` using the NeMo toolkit.

## Features
- **Data Preparation**: Convert standard transcript CSVs to NeMo-compliant `json` manifests via `scripts/prepare_data.py`.
- **Synthetic Multi-Speaker Overlap Data**: Mix single-speaker TTS voice datasets into realistic overlapping conversational datasets via `scripts/synthesize_data.py`.
- **Modular Core**: Reusable `MultitalkerASRModel` class in `src/multitalker_asr/` for integration into APIs or other scripts.
- **CLI Scripts**: Clean wrappers in the `scripts/` directory for all major operations.

---

## 1. Installation

### Prerequisites
- Python 3.10+
- `ffmpeg` installed on the system (`sudo apt install ffmpeg`).

### Install Dependencies
1. Clone or navigate to the project root.
2. Use `uv` to install the requirements from `pyproject.toml`, which includes PyTorch Lightning, Librosa, and the NeMo toolkit directly from GitHub.

```bash
uv sync
```

### Download Base Models (Recommended)
Hugging Face wrappers can sometimes hang when downloading the massive NeMo checkpoints directly in code. It's recommended to download them to a `models/` folder beforehand.

```bash
mkdir -p models
# Download Parakeet 0.6B ASR Base Model (2.3GB)
wget -c https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1/resolve/main/multitalker-parakeet-streaming-0.6b-v1.nemo -P models/

# Download Sortformer v2.1 Diarization Base Model (450MB)
wget -c https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1/resolve/main/diar_streaming_sortformer_4spk-v2.1.nemo -P models/
```

---

## 2. Data Preparation Pipeline

Multitalker ASR requires overlapping multi-speaker audio with `SegLST` JSON manifests to train effectively. We provide a two-step script process to generate this from your single-speaker TTS datasets.

### Step A: Format your single-speaker datasets
First, format your source datasets into a standard CSV `dataset.csv`:
```csv
filename,speaker_id,start_time,duration,text
audio_01.wav,nam-mien-bac,0.0,3.5,xin chào bạn
...
```

Run the preparation script to generate a single-speaker NeMo manifest:
```bash
uv run scripts/prepare_data.py \
    --csv dataset.csv \
    --audio_dir /path/to/your/audio_files/ \
    --output data/single_speaker.json
```

### Step B: Synthesize Overlapping Multi-Speaker Data
Mix the single-speaker audio into overlapping, multi-speaker conversational files:
```bash
uv run scripts/synthesize_data.py \
    --input_manifests data/single_speaker.json \
    --output_dir ./data/synthesized_train_audio/ \
    --output_manifest data/train_mixed.json \
    --num_samples 1000 \
    --max_speakers 3
```
*Note: Run this again with different inputs or random seeds to generate `data/val_mixed.json` for validation.*

---

## 3. Fine-tuning the Model

To adapt the base `Parakeet-0.6B` model to Vietnamese, run the fine-tuning script:

```bash
uv run scripts/finetune.py \
    --model_path models/multitalker-parakeet-streaming-0.6b-v1.nemo \
    --train_manifest data/train_mixed.json \
    --val_manifest data/val_mixed.json \
    --gpus 1 \
    --max_steps 10000
```
*(Note: PyTorch Lightning logs and checkpoints will be saved to the `checkpoints/` directory.)*

*(Note: If you run into `LightningModule` class validation errors during fine-tuning initialization, ensure your PyTorch Lightning version matches the exact compatibility requirements of the NeMo commit. Often `pytorch-lightning<2.0` is required).*

The script will save the newly tuned checkpoint to `models/multitalker-parakeet-streaming-0.6b-v1-finetuned.nemo`.

---

## 4. Inference

Run out-of-core inference on an audio file:

```bash
# Example running on CPU for testing
uv run scripts/infer.py \
    --model_path "models/multitalker-parakeet-streaming-0.6b-v1.nemo" \
    --audio "demo_audio.wav" \
    --output "data/transcript.json" \
    --cpu
```

*Replace `cuda=-1` and `device="cpu"` with `cuda=0` and `device="cuda"` if you are running on a machine with a 24GB+ GPU.*

### The Output format (`transcript.json`)
The output will be in NeMo's SegLST JSON format outlining exactly when each speaker spoke:
```json
[
  {
    "audio_filepath": "demo_audio.wav",
    "offset": 1.5,
    "duration": 2.1,
    "label": "speaker_0",
    "text": "xin chào các bạn"
  },
  {
    "audio_filepath": "demo_audio.wav",
    "offset": 2.8,
    "duration": 1.9,
    "label": "speaker_1",
    "text": "chào buổi sáng"
  }
]
```
