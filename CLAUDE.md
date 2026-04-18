# multitalker-asr — Project Configuration

## Overview
Vietnamese multi-speaker ASR system built on NVIDIA NeMo. Implements end-to-end training, evaluation, and inference pipelines for overlapping multi-talker speech recognition using Conformer encoder + RNNT decoder with Sortformer diarization.

## Tech Stack
- Language: Python 3.10+
- Framework: PyTorch + NVIDIA NeMo + PyTorch Lightning
- Package Manager: uv
- Test Framework: pytest (minimal coverage)
- Build Tool: hatchling
- Linter: none configured
- Python Environment: venv (`.venv/` managed by uv)

## Directory Structure
```
src/multitalker_asr/       # Main package
  configs/                 # Dataclass-based configuration (model, training, inference, eval, data)
  data/                    # Data pipeline (datasets, synthesizers, mixers, collators, manifests)
  models/                  # ASR model wrappers + speaker attribute heads
  training/                # PyTorch Lightning trainer + callbacks
  inference/               # Offline and streaming inference engines
  eval/                    # Diarization evaluation pipeline (metrics, reporters)
  utils/                   # Checkpoint, device, audio, text, format utilities
scripts/                   # CLI entry points (train, infer, eval, data prep, tokenizer)
data/                      # Training manifests, tokenizer files, corpus
models/                    # Pre-trained NeMo checkpoints
checkpoints/               # Training checkpoint outputs
tests/                     # Test suite (minimal)
docs/                      # Technical documentation
```

## Conventions
- Naming: snake_case for files, functions, variables; PascalCase for classes
- Imports: stdlib first, then third-party, then relative imports with `..` notation
- Error handling: explicit checks with early returns, loguru for logging (logger.info/warning/error/success)
- Config pattern: Python dataclasses with `@dataclass` decorator, inheriting from `BaseConfig`
- Architecture: Base class + concrete implementation pattern (BaseASRModel -> MultitalkerASRModel, BaseTrainer -> MultitalkerTrainer)
- Private attributes: underscore prefix (`self._model`, `self._trainer`)
- Type hints: used consistently with `typing` module (Optional, List, Dict, Union)
- Config management: OmegaConf for NeMo configs, dataclasses for app configs

## Commands
- Install: `uv sync`
- Dev: `uv run python scripts/train.py` (no dev server — ML training project)
- Build: `uv build`
- Test: `uv run pytest tests/`
- Lint: none configured

## Key Files
- Entry point: `src/multitalker_asr/__init__.py` (public API)
- Config: `pyproject.toml`
- Main model: `src/multitalker_asr/models/multitalker.py`
- Trainer: `src/multitalker_asr/training/trainer.py`
- Data synthesis: `src/multitalker_asr/data/synthesizers/multitalker.py`
- Training script: `scripts/train.py`
- Inference script: `scripts/infer.py`
- Full training pipeline: `scripts/run_full_training.sh`

## Important Notes
- NeMo is installed from git main branch (not PyPI): `nemo-toolkit = { git = "https://github.com/NVIDIA/NeMo.git", rev = "main" }`
- PyTorch Lightning is pinned to 1.9.5 for NeMo compatibility
- CUDA GPU required for training; CPU fallback exists for inference
- Audio data uses NeMo manifest format (JSONL with audio_filepath, text, duration fields)
- On-the-fly data synthesis generates multi-speaker mixtures during training
