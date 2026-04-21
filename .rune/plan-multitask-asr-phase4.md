# Phase 4: Curriculum Training — 3-Phase Trainer + Dynamic Loss Weighting

## Data Flow
```
CurriculumTrainer.setup(curriculum_phase=1|2|3)
  │
  ├── Phase 1 (ASR): freeze lower encoder, RNNT loss only, single-speaker data
  │     loss = L_rnnt,  freeze encoder[:18], lr=1e-4
  │
  ├── Phase 2 (Multi-talker): activate Sortformer + kernels, synthetic overlap data
  │     loss = λ₁·L_rnnt + λ₂·L_sort,  freeze encoder[:12], lr=1e-5
  │
  └── Phase 3 (Paralinguistic): activate prompt tokens + CE loss, annotated data
        loss = λ₁·L_rnnt + λ₂·L_sort + λ₃·L_prompt_ce,  lr=5e-6
        UncertaintyWeighting adjusts λ dynamically

DynamicLossWeighting(nn.Module):
  log_vars = nn.Parameter(torch.zeros(N))  # learnable uncertainty per task
  forward(losses: List[Tensor]) → weighted_sum
```

## Code Contracts

```python
# --- training/losses/dynamic_weighting.py ---
class DynamicLossWeighting(nn.Module):
    """Uncertainty-based dynamic loss weighting. Ref: Kendall et al. 2018."""
    def __init__(self, num_tasks: int): ...
    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]: ...
    # Returns (total_loss, {"task_weights": {name: float}, "task_losses": {name: float}})

# --- training/curriculum_trainer.py ---
class CurriculumPhase(str, Enum):
    ASR = "asr"                    # Phase 1
    MULTITALKER = "multitalker"    # Phase 2
    PARALINGUISTIC = "paralinguistic"  # Phase 3

class CurriculumTrainer(MultitalkerTrainer):
    """Extends MultitalkerTrainer with curriculum phase management."""
    def __init__(self, model: MultitalkerMultiTaskModel,
                 model_cfg: ModelConfig, train_cfg: TrainingConfig,
                 multitask_cfg: MultiTaskConfig): ...

    def setup(self, curriculum_phase: CurriculumPhase = CurriculumPhase.ASR) -> None:
        """Configure model freezing, loss weights, data loaders per phase."""

    def _configure_phase_asr(self) -> None: ...
    def _configure_phase_multitalker(self) -> None: ...
    def _configure_phase_paralinguistic(self) -> None: ...
    def _freeze_encoder_layers(self, num_frozen: int) -> None: ...
    def _setup_multitask_data_loaders(self) -> None: ...

# --- scripts/train_multitask.py ---
# CLI entry point with --curriculum_phase argument
```

## Tasks

### Wave 1 (independent)

#### Task 1a: DynamicLossWeighting module
- **File**: `src/multitalker_asr/training/losses/dynamic_weighting.py` — new
- **touches**: [training/losses/dynamic_weighting.py, training/losses/__init__.py]
- **provides**: [DynamicLossWeighting]
- **requires**: []
- **Logic**: Learnable `log_var` per task (nn.Parameter). Loss = `Σ (exp(-log_var_i) * loss_i + log_var_i)`. This auto-balances task magnitudes. Return total + per-task weights/losses for logging.
- **Edge cases**: Single-task mode (Phase 1) → pass through without weighting. Loss is NaN → clamp log_var, log warning.

#### Task 1b: CurriculumPhase enum + TrainingConfig extension
- **File**: `src/multitalker_asr/configs/training.py` — edit
- **touches**: [configs/training.py]
- **provides**: [CurriculumPhase enum, extended TrainingConfig fields]
- **requires**: []
- **Logic**: Add to TrainingConfig: `curriculum_phase: str = "asr"`, `num_frozen_layers: int = 18`, `use_dynamic_weighting: bool = True`, `multitask_train_manifest: Optional[str] = None` (annotated manifest for Phase 3).

### Wave 2 (depends on Wave 1)

#### Task 2a: CurriculumTrainer class
- **File**: `src/multitalker_asr/training/curriculum_trainer.py` — new
- **touches**: [training/curriculum_trainer.py, training/__init__.py]
- **provides**: [CurriculumTrainer]
- **requires**: [MultitalkerTrainer, MultitalkerMultiTaskModel, DynamicLossWeighting, CurriculumPhase, MultitaskStreamingDataset]
- **depends_on**: [task-1a, task-1b]
- **Logic**:
  1. `setup()`: Call super().setup() then apply phase-specific config
  2. `_configure_phase_asr()`: Freeze encoder layers 0..17, disable prompt loss, set lr=1e-4, use single-speaker manifest
  3. `_configure_phase_multitalker()`: Freeze layers 0..11, enable Sortformer, set lr=1e-5, use synthetic multi-speaker data
  4. `_configure_phase_paralinguistic()`: Unfreeze prompt embed + classifier, enable CE loss, set lr=5e-6, use multitask_train_manifest, init DynamicLossWeighting
  5. `_freeze_encoder_layers()`: Iterate encoder.layers[:n], set requires_grad=False
  6. `_setup_multitask_data_loaders()`: Use DataLoaderFactory.create_multitask_dataloader()
  7. Override `train()` to use forward_multitask() instead of forward()
- **Edge cases**: Resume from Phase 2 checkpoint into Phase 3 → load checkpoint, reconfigure freezing. Encoder has fewer layers than num_frozen → freeze all available, log warning.

### Wave 3 (CLI + tests)

#### Task 3a: CLI entry point
- **File**: `scripts/train_multitask.py` — new
- **touches**: [scripts/train_multitask.py]
- **provides**: [CLI for curriculum training]
- **requires**: [CurriculumTrainer, MultitalkerMultiTaskModel]
- **depends_on**: [task-2a]
- **Logic**: Extend train.py argparse with: `--curriculum_phase`, `--num_frozen_layers`, `--use_dynamic_weighting`, `--multitask_train_manifest`, `--prompt_ce_weight`. Create MultitalkerMultiTaskModel + CurriculumTrainer, call train_and_save().

#### Task 3b: Unit tests
- **File**: `tests/test_curriculum_trainer.py` — new
- **touches**: [tests/test_curriculum_trainer.py]
- **depends_on**: [task-2a]
- **Tests**:
  - DynamicLossWeighting: 2 losses → weighted sum, weights are learnable
  - DynamicLossWeighting: single loss pass-through
  - CurriculumPhase enum values
  - _freeze_encoder_layers: correct params frozen (mock encoder)
  - Phase config: ASR phase disables prompt loss
  - Phase config: Paralinguistic phase enables all losses

## Failure Scenarios

| When | Then | Error |
|------|------|-------|
| Phase 3 started without Phase 1/2 checkpoint | Untrained encoder, bad results | Log warning, allow but advise against |
| Encoder has <18 layers (e.g., small model) | IndexError on freeze | Clamp to min(num_frozen, total_layers), log |
| Dynamic weighting log_var diverges | One task dominates | Clamp log_var to [-6, 6] range |
| multitask_train_manifest not set for Phase 3 | No task labels in data | Raise ValueError: "Phase 3 requires --multitask_train_manifest" |

## Rejection Criteria
- DO NOT train all tasks simultaneously from the start — curriculum is mandatory
- DO NOT hardcode learning rates — read from config, override per phase
- DO NOT skip DynamicLossWeighting in Phase 3 — static weights cause gradient interference
- DO NOT modify MultitalkerTrainer — extend via inheritance

## Cross-Phase Context
- **Assumes from Phase 1**: MultiTaskConfig exists
- **Assumes from Phase 2**: MultitalkerMultiTaskModel.forward_multitask() works, returns loss dict
- **Assumes from Phase 3**: DataLoaderFactory.create_multitask_dataloader() works
- **Exports for Phase 6**: Trained checkpoints — inference engine loads these

## Acceptance Criteria
- `uv run pytest tests/test_curriculum_trainer.py` passes
- DynamicLossWeighting gradients flow to log_var parameters
- Phase configs correctly freeze/unfreeze expected layers
- `scripts/train_multitask.py --help` shows all curriculum args
- Trainer can be instantiated with each of the 3 phases without error
