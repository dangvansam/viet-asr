# Developer Guide: multitalker-asr

## What This Does
A Vietnamese multi-speaker automatic speech recognition system that can transcribe overlapping speech from multiple talkers simultaneously. Built on NVIDIA NeMo with Conformer encoder, RNNT decoder, and Sortformer diarization.

## Quick Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (requires uv)
uv sync

# Download pre-trained models (see README.md for model paths)
# Models go in models/ directory

# Run training (full pipeline)
bash scripts/run_full_training.sh

# Run training (direct script)
uv run python scripts/train.py \
  --asr_model models/multitalker-parakeet-streaming-0.6b-v1.nemo \
  --train_manifest data/train_single_speaker.json \
  --val_manifest data/val_single_speaker.json

# Run inference
uv run python scripts/infer.py --audio_path demo.wav

# Run tests
uv run pytest tests/
```

## Key Files
- `src/multitalker_asr/__init__.py` — Public API, all exported classes
- `src/multitalker_asr/models/multitalker.py` — Main ASR model wrapper (loading, saving, inference)
- `src/multitalker_asr/training/trainer.py` — PyTorch Lightning training orchestration
- `src/multitalker_asr/data/synthesizers/multitalker.py` — On-the-fly multi-speaker audio mixing
- `src/multitalker_asr/configs/training.py` — Training configuration dataclass
- `scripts/train.py` — CLI training entry point
- `scripts/infer.py` — CLI inference entry point
- `scripts/run_full_training.sh` — End-to-end training pipeline script
- `pyproject.toml` — Project metadata, dependencies, build config

## How to Contribute
1. Fork or branch from main
2. Make changes, run tests: `uv run pytest tests/`
3. Open a PR — describe what and why

## Common Issues
- **ModuleNotFoundError** — Virtual environment not activated. Run: `source .venv/bin/activate`
- **ImportError: cannot import name X** — Dependencies outdated. Run: `uv sync`
- **CUDA out of memory** — Reduce `batch_size` in training config or script arguments
- **NeMo model download fails** — Check network connectivity; pre-trained models require Hugging Face access
- **`nemo-toolkit` build fails** — NeMo is installed from git main; ensure you have `gcc`, `g++`, and CUDA toolkit installed

## Who to Ask
Check `git log --format='%an' | sort | uniq -c | sort -rn` for active contributors.
