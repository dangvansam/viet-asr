# Phase 3: Data Pipeline — Extended Dataset + Collator + Manifest

## Data Flow
```
JSONL Manifest (extended format):
  {"audio_filepath": "x.wav", "text": "xin chao", "duration": 3.2,
   "emotion": "happy", "gender": "male", "age": "young",
   "voice_state": "sober", "language": "vi"}
       ↓
MultitaskStreamingDataset.__iter__()
  1. Sample N utterances (2..max_speakers)
  2. Mix audio via MultiTalkerMixer → mixed_audio, supervisions
  3. Map string labels → integer IDs via TaskTokenRegistry
  4. Yield: {audio, text_ids, task_labels: {emotion: int, gender: int, ...}, masks}
       ↓
MultitaskCollator.__call__(batch)
  1. Pad audio, text, masks
  2. Stack task_labels into {task: Tensor[B]}
  3. Return (audio, audio_lens, text, text_lens, task_labels, spk_mask, bg_mask)
       ↓
DataLoaderFactory.create_multitask_dataloader() → DataLoader
```

## Code Contracts

```python
# --- data/datasets/multitask.py ---
class MultitaskStreamingDataset(StreamingMultitalkerDataset):
    """Extends StreamingMultitalkerDataset with paralinguistic label loading."""
    def __init__(
        self,
        manifest_paths: List[str],
        tokenizer=None,
        mixer: Optional[MultiTalkerMixer] = None,
        task_registry: Optional[TaskTokenRegistry] = None,
        max_speakers: int = 2,
        default_labels: Optional[Dict[str, str]] = None,  # fallback for missing fields
        **kwargs,
    ): ...

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Yields dicts with 'task_labels' key added to base class output."""

    def _parse_task_labels(self, utterance: dict) -> Dict[str, int]:
        """Extract task labels from manifest entry, map to int via registry."""

# --- data/collators/multitask.py ---
class MultitaskCollator(MultitalkerCollator):
    """Extends MultitalkerCollator to batch task_labels."""
    def __call__(self, batch: List[dict]) -> Tuple:
        """Returns: (audio, audio_lens, text, text_lens, task_labels, spk_mask, bg_mask)
        where task_labels is Dict[str, Tensor[B]]"""
```

## Tasks

### Wave 1 (independent)

#### Task 1a: MultitaskStreamingDataset
- **File**: `src/multitalker_asr/data/datasets/multitask.py` — new
- **touches**: [data/datasets/multitask.py, data/datasets/__init__.py]
- **provides**: [MultitaskStreamingDataset]
- **requires**: [StreamingMultitalkerDataset, TaskTokenRegistry]
- **Logic**:
  1. Inherit from StreamingMultitalkerDataset
  2. In `__iter__`: call super().__iter__() pattern but override to add task label parsing
  3. `_parse_task_labels()`: Read fields (emotion, gender, age, voice_state, language) from manifest entry. Map string→int via TaskTokenRegistry. Missing field → use default_labels or 0.
  4. For multi-speaker mix: pick task labels from the **primary speaker** (first speaker in arrival order, consistent with Sortformer ATS convention)
  5. Add `task_labels` dict to yielded sample
- **Edge cases**: Manifest entry missing "emotion" field → use default. Unknown label string (e.g. "excited") → log warning, use NEUTRAL(3). Empty manifest → raise ValueError.

#### Task 1b: MultitaskCollator
- **File**: `src/multitalker_asr/data/collators/multitask.py` — new
- **touches**: [data/collators/multitask.py, data/collators/__init__.py]
- **provides**: [MultitaskCollator]
- **requires**: [MultitalkerCollator]
- **Logic**:
  1. Inherit from MultitalkerCollator
  2. In `__call__`: call super() for audio/text/mask collation
  3. Extract task_labels from each sample, stack per-task into Tensor[B]
  4. Return extended tuple with task_labels dict
- **Edge cases**: Sample missing task_labels key → fill with zeros. Inconsistent task keys across batch → union of all keys, fill missing with 0.

### Wave 2 (depends on Wave 1)

#### Task 2a: Update DataLoaderFactory
- **File**: `src/multitalker_asr/data/factory.py` — edit
- **touches**: [data/factory.py]
- **provides**: [DataLoaderFactory.create_multitask_dataloader()]
- **requires**: [MultitaskStreamingDataset, MultitaskCollator]
- **depends_on**: [task-1a, task-1b]
- **Logic**: Add static method `create_multitask_dataloader()` that creates MultitaskStreamingDataset + MultitaskCollator. Same signature as existing methods but with `task_registry` and `default_labels` params.

#### Task 2b: Update exports
- **File**: `src/multitalker_asr/data/__init__.py` — edit
- **File**: `src/multitalker_asr/__init__.py` — edit
- **touches**: [data/__init__.py, __init__.py]
- **depends_on**: [task-1a, task-1b, task-2a]

### Wave 3 (tests)

#### Task 3a: Unit tests
- **File**: `tests/test_multitask_data.py` — new
- **touches**: [tests/test_multitask_data.py]
- **depends_on**: [task-2a]
- **Tests**:
  - Create sample JSONL manifest with all task fields → dataset yields correct task_labels
  - Manifest with missing fields → default values used, no crash
  - MultitaskCollator: task_labels are Tensor[B] after collation
  - DataLoaderFactory.create_multitask_dataloader() returns functional DataLoader
  - Multi-speaker mix: task_labels come from primary speaker

#### Task 3b: Sample manifest fixture
- **File**: `tests/fixtures/sample_multitask_manifest.json` — new
- **touches**: [tests/fixtures/]
- **Logic**: 5-10 JSONL entries with all fields populated. Include edge cases: missing emotion, unknown gender value.

## Failure Scenarios

| When | Then | Error |
|------|------|-------|
| Manifest has no task label fields at all | All defaults used | Log warning: "No task labels found, using defaults" |
| Unknown label string ("excited" for emotion) | Mapped to default (NEUTRAL) | Log warning with the unknown value |
| Mixed batch: some samples have labels, some don't | Inconsistent tensor shapes | Fill missing with default (0) before stacking |
| Audio file referenced in manifest doesn't exist | Loader crash | Caught by existing base class error handling |

## Rejection Criteria
- DO NOT load full audio in _parse_task_labels — only parse JSON fields
- DO NOT create a new manifest format — extend existing JSONL with optional fields
- DO NOT break backward compatibility — old manifests without task fields must still work
- DO NOT couple task label parsing to specific NeMo data pipeline — keep it independent

## Cross-Phase Context
- **Assumes from Phase 1**: TaskTokenRegistry exists with get_embed_id() and task_names()
- **Assumes**: StreamingMultitalkerDataset at data/datasets/streaming.py, MultitalkerCollator at data/collators/multitalker.py
- **Exports for Phase 4**: MultitaskStreamingDataset, MultitaskCollator, DataLoaderFactory.create_multitask_dataloader() — trainer uses these
- **Manifest format**: Extended JSONL with optional fields: emotion, gender, age, voice_state, language. Old manifests work (defaults applied).

## Acceptance Criteria
- `uv run pytest tests/test_multitask_data.py` passes
- Old-format manifest (no task fields) loads without error
- New-format manifest produces correct task_labels dict
- Collated batch task_labels has shape {task_name: Tensor[B]} for all 6 tasks
- DataLoader iteration produces valid batches (no crashes over 10 iterations)
