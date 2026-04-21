# Phase 2: Multi-Task Model — Encoder Wrapper + Prompt Prepend + Dual Loss

## Data Flow
```
task_labels: Dict[str, Tensor[B]]  →  PromptEmbedding  →  prompt_embeds [B, 6, D]
speech_features: Tensor[B, T, D]   ─────────────────────→  concat  →  [B, T+6, D]
                                                               ↓
                                                    FastConformer Encoder
                                                               ↓
                                                    encoder_out [B, T'+6, H]
                                                         ╱          ╲
                                              [:, :6, :]           [:, 6:, :]
                                                 ↓                     ↓
                                           CE Loss (prompt)     RNNT Loss (ASR)
                                                 ↓                     ↓
                                            loss_prompt            loss_rnnt
                                                 └────── L_total ──────┘
                                        λ₁·loss_rnnt + λ₃·loss_prompt
```

## Code Contracts

```python
# --- models/multitask_model.py ---
class MultitalkerMultiTaskModel(BaseASRModel):
    """Wraps existing MultitalkerASRModel + adds prompt-based multi-task capability."""
    def __init__(self, model_cfg: ModelConfig, multitask_cfg: MultiTaskConfig):
        self._base_model: MultitalkerASRModel  # existing model
        self._prompt_embed: PromptEmbedding
        self._prompt_classifier: nn.Linear  # encoder_hidden → total_prompt_tokens
        self._multitask_cfg: MultiTaskConfig

    def load_models(self) -> None:
        """Load ASR + diar models, then init prompt layers."""

    def forward_multitask(
        self,
        audio: torch.Tensor,           # [B, samples]
        audio_lengths: torch.Tensor,    # [B]
        task_labels: Dict[str, torch.Tensor],  # {task: [B]}
        text: torch.Tensor = None,      # [B, L] for RNNT
        text_lengths: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Returns {"loss_total", "loss_rnnt", "loss_prompt", "prompt_preds"}"""

    def _prepend_prompts(
        self, speech: torch.Tensor, task_labels: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, int]:
        """Prepend prompt embeddings to speech features. Returns (augmented, num_prepended)."""

    def _compute_prompt_loss(
        self, encoder_out: torch.Tensor, task_labels: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """CE loss on first 6 encoder positions. Ref: sense_voice/model.py:793-807"""

# --- training/losses/__init__.py --- (new directory)
# --- training/losses/multi_task_loss.py ---
class MultiTaskLoss(nn.Module):
    """Combines RNNT loss + prompt CE loss with configurable weights."""
    def __init__(self, rnnt_weight: float = 1.0, prompt_weight: float = 1.0): ...
    def forward(self, loss_rnnt, loss_prompt) -> torch.Tensor: ...
```

## Tasks

### Wave 1 (foundation)

#### Task 1a: training/losses/ directory + MultiTaskLoss
- **File**: `src/multitalker_asr/training/losses/__init__.py` — new
- **File**: `src/multitalker_asr/training/losses/multi_task_loss.py` — new
- **touches**: [training/losses/__init__.py, training/losses/multi_task_loss.py]
- **provides**: [MultiTaskLoss]
- **requires**: []
- **Logic**: Weighted sum `λ₁ * loss_rnnt + λ₃ * loss_prompt`. Both weights configurable. Returns dict with total + individual losses for logging.
- **Edge cases**: NaN loss → log warning, replace with 0. Negative weights → ValueError.

### Wave 2 (depends on Wave 1 + Phase 1 exports)

#### Task 2a: MultitalkerMultiTaskModel
- **File**: `src/multitalker_asr/models/multitask_model.py` — new
- **touches**: [models/multitask_model.py]
- **provides**: [MultitalkerMultiTaskModel]
- **requires**: [MultitalkerASRModel, PromptEmbedding, TaskTokenRegistry, MultiTaskLoss]
- **depends_on**: [task-1a]
- **Logic**:
  1. `__init__`: Create base model, prompt embedding, prompt classifier (Linear(encoder_hidden, total_tokens))
  2. `load_models()`: Delegate to base model, then freeze/unfreeze based on curriculum phase
  3. `_prepend_prompts()`: Get prompt embeds [B,6,D], concat with speech [B,T,D] → [B,T+6,D]. Adjust lengths += 6. Follow SenseVoice model.py:744-774.
  4. `_compute_prompt_loss()`: Take encoder_out[:,:6,:], project via prompt_classifier, compute CE loss per task position against task_labels. Follow SenseVoice model.py:793-807.
  5. `forward_multitask()`: Extract fbank → prepend prompts → encode → split output → RNNT loss on speech part + CE loss on prompt part → combine via MultiTaskLoss
- **Edge cases**: encoder_hidden size mismatch with prompt_classifier → detect in load_models(). If task_labels is None (inference mode) → skip CE loss, return preds only.
- **IMPORTANT**: Access encoder via `self._base_model.asr_model.encoder`. Do NOT modify NeMo source. Wrap the encode call.

#### Task 2b: Encoder source support (pretrained/scratch/funasr)
- **File**: `src/multitalker_asr/models/multitask_model.py` — append to Task 2a
- **touches**: [models/multitask_model.py]
- **provides**: [load_encoder_from_funasr(), load_encoder_from_nemo()]
- **requires**: [MultitalkerMultiTaskModel]
- **depends_on**: [task-2a]
- **Logic**: Methods to extract encoder weights from FunASR checkpoint state_dict and map to NeMo encoder. Key mapping: inspect `audio_encoder` keys from FunASR model, remap to NeMo `encoder.*` keys. Log mismatched/missing keys.
- **Edge cases**: Key mismatch → log warning, load with strict=False. Dimension mismatch → raise RuntimeError with details.

### Wave 3 (exports + tests)

#### Task 3a: Update exports
- **File**: `src/multitalker_asr/models/__init__.py` — edit
- **File**: `src/multitalker_asr/__init__.py` — edit
- **touches**: [models/__init__.py, __init__.py]
- **provides**: [public exports for MultitalkerMultiTaskModel, MultiTaskLoss]
- **depends_on**: [task-2a, task-2b]

#### Task 3b: Unit tests
- **File**: `tests/test_multitask_model.py` — new
- **touches**: [tests/test_multitask_model.py]
- **depends_on**: [task-2a]
- **Tests**:
  - `_prepend_prompts()`: output shape [B, T+6, D], lengths adjusted
  - `_compute_prompt_loss()`: returns scalar loss, per-task accuracy dict
  - `MultiTaskLoss`: weighted combination correct, NaN handling
  - Mock encoder test: forward_multitask returns expected keys
  - Verify prompt classifier output dim matches total_prompt_tokens

## Failure Scenarios

| When | Then | Error |
|------|------|-------|
| Encoder hidden dim != prompt embed dim | Cat fails at prepend | Detect in __init__, add projection layer if mismatched |
| NeMo model doesn't expose encoder | Can't prepend | Access via `model.encoder` (NeMo EncDecRNNTBPEModel guarantees this) |
| FunASR state_dict keys don't match NeMo | Weights not loaded | strict=False + log missing keys, user must verify |
| RNNT loss returns None (no text labels) | Total loss is NaN | Guard: if text is None, loss_rnnt = 0 |

## Rejection Criteria
- DO NOT modify NeMo source code — wrap, don't patch
- DO NOT put training loop logic here — this is model architecture only
- DO NOT skip the prompt classifier projection — encoder output dim likely != embed dim
- DO NOT hardcode encoder hidden size — read from loaded model dynamically

## Cross-Phase Context
- **Assumes from Phase 1**: MultiTaskConfig, PromptEmbedding, TaskTokenRegistry exist and tested
- **Assumes**: MultitalkerASRModel exists at `models/multitalker.py` with `asr_model` property
- **Exports for Phase 4**: MultitalkerMultiTaskModel.forward_multitask() — trainer calls this
- **Exports for Phase 5**: MultitalkerMultiTaskModel — Phase 5 extends vocab on its RNNT decoder
- **Exports for Phase 6**: MultitalkerMultiTaskModel — inference engine wraps this

## Acceptance Criteria
- `uv run pytest tests/test_multitask_model.py` passes
- `_prepend_prompts(speech[2,100,80], labels)` returns [2, 106, 80]
- `_compute_prompt_loss()` returns scalar loss + per-task accuracy dict
- `forward_multitask()` returns dict with keys: loss_total, loss_rnnt, loss_prompt, prompt_preds
- Model can be instantiated with encoder_source="nemo" and "scratch"
