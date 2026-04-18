# Project Contract

## contract.code

### Python Safety
- **No bare `except:`** — always catch specific exceptions or use `except Exception`
- **No mutable default arguments** — use `None` with `or` pattern (already followed)
- **Type hints required** on all public function signatures

### Logging
- **Use `loguru.logger`** — never stdlib `logging` or `print()` for operational output
- **No `print()` in library code** (`src/multitalker_asr/`) — scripts may use print for CLI output

### Imports
- **No wildcard imports** (`from module import *`)
- **Relative imports** within `src/multitalker_asr/` package

### Configuration
- **Dataclass configs** — new configuration must use `@dataclass` inheriting `BaseConfig`
- **OmegaConf mutations** — always use `open_dict` context manager

## contract.data

### Audio Data
- **NeMo manifest format** — JSONL with required fields: `audio_filepath`, `text`, `duration`
- **Sample rate** — 16kHz unless explicitly configured otherwise
- **Lhotse CutSet** — use for multi-speaker data with supervision segments

## contract.testing

- **Tests in `tests/`** — not co-located with source
- **No test data in git** — use fixtures or generate programmatically

## contract.dependencies

- **NeMo from git main** — do not switch to PyPI release without explicit decision
- **PyTorch Lightning pinned to 1.9.5** — NeMo compatibility constraint
- **UV for dependency management** — do not add requirements.txt or pip-based workflows
