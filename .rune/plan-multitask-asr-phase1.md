# Phase 1: Foundation — Configs + Task Tokens + Prompt Embedding

## Data Flow
```
MultiTaskConfig (dataclass)
  ├── task_tokens: dict → TaskTokenRegistry (enum + ID mapping)
  │     emotion: [HAPPY=0, SAD=1, ANGRY=2, NEUTRAL=3, FEAR=4, DISGUST=5, SURPRISE=6]
  │     gender:  [MALE=0, FEMALE=1]
  │     age:     [CHILD=0, YOUNG=1, MIDDLE_AGE=2, OLD=3]
  │     voice_state: [SOBER=0, DRUNK=1]
  │     language: [VI=0, EN=1, ZH=2, AUTO=3]
  │     textnorm: [WITH_ITN=0, WITHOUT_ITN=1]
  │
  └── PromptEmbedding(nn.Module)
        self.embed = nn.Embedding(total_tokens, input_size)
        forward(task_labels) → [B, num_prompt_positions, input_size]
```

## Code Contracts

```python
# --- configs/multitask.py ---
@dataclass
class MultiTaskConfig(BaseConfig):
    num_prompt_positions: int = 6  # lang, emotion, gender, age, voice_state, textnorm
    prompt_embed_dim: int = 80     # must match encoder input_size (fbank dim)
    emotion_classes: int = 7
    gender_classes: int = 2
    age_classes: int = 4
    voice_state_classes: int = 2
    language_classes: int = 4
    textnorm_classes: int = 2
    ce_loss_weight: float = 1.0    # weight for prompt CE loss
    encoder_source: str = "nemo"   # "nemo" | "funasr" | "scratch"

# --- models/prompt_embedding.py ---
class TaskTokenRegistry:
    """Maps task names to token ID ranges. Each task gets a contiguous block."""
    def __init__(self, config: MultiTaskConfig): ...
    def get_embed_id(self, task: str, class_idx: int) -> int: ...
    def total_tokens(self) -> int: ...
    def task_names(self) -> List[str]: ...

class PromptEmbedding(nn.Module):
    """SenseVoice-style prompt token embedding. Ref: sense_voice/model.py:642-648"""
    def __init__(self, config: MultiTaskConfig): ...
    def forward(self, task_labels: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args: task_labels: {task_name: tensor[B]} with class indices per task
        Returns: prompt_embeds [B, num_prompt_positions, embed_dim]
        """
```

## Tasks

### Wave 1 (independent)

#### Task 1a: MultiTaskConfig dataclass
- **File**: `src/multitalker_asr/configs/multitask.py` — new
- **touches**: [configs/multitask.py, configs/__init__.py]
- **provides**: [MultiTaskConfig]
- **requires**: [BaseConfig from configs/base.py]
- **Logic**: Dataclass with fields for all task class counts, prompt dimensions, loss weight, encoder source. Inherit from BaseConfig. Add `total_prompt_tokens` property that sums all class counts.
- **Edge cases**: Validate embed_dim > 0, all class counts > 0 in `__post_init__`

#### Task 1b: TaskTokenRegistry
- **File**: `src/multitalker_asr/models/prompt_embedding.py` — new
- **touches**: [models/prompt_embedding.py]
- **provides**: [TaskTokenRegistry]
- **requires**: [MultiTaskConfig]
- **Logic**: Assign contiguous ID blocks: language=[0..3], emotion=[4..10], gender=[11..12], age=[13..16], voice_state=[17..18], textnorm=[19..20]. `get_embed_id("emotion", 2)` returns 6. Store as ordered dict.
- **Edge cases**: Invalid task name → ValueError. Class index out of range → ValueError.

### Wave 2 (depends on Wave 1)

#### Task 2a: PromptEmbedding module
- **File**: `src/multitalker_asr/models/prompt_embedding.py` — append
- **touches**: [models/prompt_embedding.py]
- **provides**: [PromptEmbedding]
- **requires**: [TaskTokenRegistry, MultiTaskConfig]
- **depends_on**: [task-1b]
- **Logic**: `nn.Embedding(registry.total_tokens, config.prompt_embed_dim)`. In forward(): for each task in fixed order, look up embed ID from task_labels dict, gather embeddings, stack to [B, 6, embed_dim]. Follow SenseVoice pattern (model.py:744-774).
- **Edge cases**: Missing task key in task_labels → use default (0). Batch dim mismatch → RuntimeError.

#### Task 2b: Export in __init__.py
- **File**: `src/multitalker_asr/configs/__init__.py` — edit
- **File**: `src/multitalker_asr/models/__init__.py` — edit
- **File**: `src/multitalker_asr/__init__.py` — edit
- **touches**: [configs/__init__.py, models/__init__.py, __init__.py]
- **provides**: [public exports]
- **requires**: [MultiTaskConfig, TaskTokenRegistry, PromptEmbedding]
- **depends_on**: [task-1a, task-2a]

### Wave 3 (tests)

#### Task 3a: Unit tests
- **File**: `tests/test_prompt_embedding.py` — new
- **touches**: [tests/test_prompt_embedding.py]
- **requires**: [MultiTaskConfig, TaskTokenRegistry, PromptEmbedding]
- **depends_on**: [task-2a, task-2b]
- **Tests**:
  - TaskTokenRegistry: correct ID mapping, total count, ValueError on invalid task/index
  - PromptEmbedding: output shape [B, 6, 80], gradient flows, missing task key uses default
  - MultiTaskConfig: validation rejects invalid values, total_prompt_tokens correct

## Failure Scenarios

| When | Then | Error |
|------|------|-------|
| embed_dim doesn't match encoder input_size | Shape mismatch in Phase 2 concat | Validate in MultiTaskConfig or at model init |
| Task label tensor has wrong batch size | Embedding lookup fails | Check B matches in PromptEmbedding.forward() |
| Unknown task name passed to registry | Silent wrong ID | Raise ValueError with valid task names |

## Rejection Criteria
- DO NOT hardcode token IDs as magic numbers — use TaskTokenRegistry
- DO NOT use separate nn.Embedding per task — use single shared embedding (SenseVoice pattern)
- DO NOT add training logic here — this phase is data structures only
- DO NOT import NeMo or FunASR — this phase has zero external ML dependencies

## Cross-Phase Context
- **Assumes**: BaseConfig exists at `configs/base.py` (verified in scout)
- **Exports for Phase 2**: MultiTaskConfig, TaskTokenRegistry, PromptEmbedding — Phase 2 uses these to build the multi-task model wrapper
- **Exports for Phase 3**: MultiTaskConfig.task_names() — Phase 3 uses this to validate manifest fields

## Acceptance Criteria
- `uv run pytest tests/test_prompt_embedding.py` passes
- PromptEmbedding(default_config).forward(sample_labels).shape == [2, 6, 80]
- TaskTokenRegistry correctly maps all 6 tasks with no ID overlap
- MultiTaskConfig rejects embed_dim=0 with ValueError
