# Project Conventions

## Naming
- **Files**: snake_case (e.g., `multitalker.py`, `streaming.py`, `checkpoint.py`)
- **Classes**: PascalCase (e.g., `MultitalkerASRModel`, `TrainingConfig`, `DataLoaderFactory`)
- **Functions/methods**: snake_case (e.g., `load_models`, `_configure_datasets`)
- **Private members**: underscore prefix (e.g., `self._model`, `self._trainer`, `self._train_cfg`)
- **Constants/Enums**: UPPER_CASE enum values (e.g., `TrainingMode.FINETUNE`)

## Import Style
- stdlib → third-party → relative, separated by blank lines
- Relative imports with `..` notation for intra-package (e.g., `from ..configs import ModelConfig`)
- Named imports preferred over wildcard
- Lazy imports for heavy dependencies (e.g., `from nemo.collections.common.tokenizers import SentencePieceTokenizer` inside methods)

## Architecture Pattern
- **Base + Implementation**: Abstract base class defines interface, concrete class implements
  - `BaseASRModel` → `MultitalkerASRModel`
  - `BaseTrainer` → `MultitalkerTrainer`
  - `BaseConfig` → `TrainingConfig`, `ModelConfig`, etc.
  - `BaseHead` → `SpeakerHead`, `GenderHead`, etc.
- **Factory pattern**: `DataLoaderFactory`, `ConfigFactory` for object creation
- **Manager pattern**: `DeviceManager`, `CheckpointManager` for resource management

## Configuration
- App configs: Python `@dataclass` inheriting from `BaseConfig`
- NeMo configs: `OmegaConf` with `open_dict` context manager for mutation
- Config fields have sensible defaults; override via constructor or argparse
- Enums used for mode selection (e.g., `TrainingMode`)

## Error Handling
- `loguru.logger` for all logging (not stdlib `logging`)
- Pattern: `logger.info()` for progress, `logger.warning()` for fallbacks, `logger.error()` for failures, `logger.success()` for completion
- Early return on validation failure rather than deep nesting
- `RuntimeError` for precondition violations (e.g., model not loaded)
- `ValueError` for invalid inputs

## Data Pipeline
- NeMo manifest format: JSONL with `audio_filepath`, `text`, `duration` fields
- Lhotse CutSet for multi-speaker data with supervision segments
- On-the-fly synthesis via `MultitalkerSynthesizer` and `StreamingMultitalkerDataset`

## Testing
- Tests in `tests/` directory (separate from source)
- Minimal test coverage currently
- No linter or formatter configured
