# Phase 6: Streaming Multi-Task Inference + Evaluation

## Data Flow
```
Audio (16kHz) → MultitaskInferenceEngine
  │
  ├── Streaming Path:
  │     CacheAwareStreamingAudioBuffer → chunks
  │       → Sortformer → speaker_activity [T×N]
  │       → Per-speaker: prompt + encode + RNNT decode
  │       → Raw transcript per speaker (with <EOU> markers)
  │       → Prompt position argmax → paralinguistic labels per speaker
  │
  ├── Offline Post-proc (optional):
  │     Raw transcripts + audio → FunASRPostProcessor
  │       → ITN + PnC refined text per speaker
  │
  └── Output: List[SpeakerResult]
        SpeakerResult:
          speaker_id: str
          text: str              # raw or refined
          text_refined: str      # ITN + PnC (if post-proc enabled)
          emotion: str           # "happy", "sad", ...
          gender: str            # "male", "female"
          age: str               # "child", "young", ...
          voice_state: str       # "sober", "drunk"
          language: str          # "vi", "en", ...
          eou_detected: bool
          timestamps: List[Dict] # word-level if available
```

## Code Contracts

```python
# --- inference/multitask.py ---
@dataclass
class SpeakerResult:
    speaker_id: str
    text: str
    text_refined: Optional[str] = None
    emotion: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[str] = None
    voice_state: Optional[str] = None
    language: Optional[str] = None
    eou_detected: bool = False
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 0.0

class MultitaskInferenceEngine(BaseInferenceEngine):
    """End-to-end multi-task inference with streaming + post-processing."""
    def __init__(
        self,
        model: MultitalkerMultiTaskModel,
        inference_cfg: InferenceConfig,
        multitask_cfg: MultiTaskConfig,
        post_processor: Optional[FunASRPostProcessor] = None,
    ): ...

    def infer(self, audio_path: str) -> List[SpeakerResult]:
        """Full pipeline: diarize → transcribe → classify → refine."""

    def infer_streaming(
        self, audio_path: str, chunk_callback: Optional[Callable] = None
    ) -> List[SpeakerResult]:
        """Streaming inference with per-chunk callback."""

    def _decode_prompt_labels(
        self, encoder_out: torch.Tensor, task_registry: TaskTokenRegistry
    ) -> Dict[str, str]:
        """Argmax on prompt positions → string labels."""

    def _detect_eou(self, token_ids: List[int], eou_token_id: int) -> bool:
        """Check if <EOU> token appears in decoded sequence."""

# --- eval/multitask_evaluator.py ---
class MultitaskEvaluator:
    """Evaluates multi-task ASR: WER/CER + paralinguistic accuracy."""
    def __init__(self, engine: MultitaskInferenceEngine): ...

    def evaluate(
        self, manifest_path: str, output_dir: str
    ) -> Dict[str, float]:
        """Returns {wer, cer, emotion_acc, gender_acc, age_acc, voice_state_acc, eou_f1}"""

    def _compute_asr_metrics(self, predictions, references) -> Dict: ...
    def _compute_task_accuracy(self, task, predictions, references) -> float: ...
    def _generate_report(self, metrics: Dict, output_dir: str) -> str: ...
```

## Tasks

### Wave 1 (data structures)

#### Task 1a: SpeakerResult dataclass
- **File**: `src/multitalker_asr/inference/multitask.py` — new
- **touches**: [inference/multitask.py]
- **provides**: [SpeakerResult]
- **requires**: []
- **Logic**: Frozen dataclass with all output fields. Add `to_dict()` method for JSON serialization. Add `__str__()` for readable console output.

### Wave 2 (inference engine)

#### Task 2a: MultitaskInferenceEngine
- **File**: `src/multitalker_asr/inference/multitask.py` — append
- **touches**: [inference/multitask.py, inference/__init__.py]
- **provides**: [MultitaskInferenceEngine]
- **requires**: [SpeakerResult, MultitalkerMultiTaskModel, FunASRPostProcessor, TaskTokenRegistry]
- **depends_on**: [task-1a]
- **Logic**:
  1. `infer()`: Load audio → diarize (Sortformer) → per-speaker: create dummy task_labels (zeros for inference) → prepend prompts → encode → RNNT decode → decode prompt labels → optionally refine with FunASR → build SpeakerResult
  2. `_decode_prompt_labels()`: Take encoder_out[:, :6, :], project via prompt_classifier, argmax per position, map int→string via TaskTokenRegistry reverse lookup
  3. `_detect_eou()`: Check if EOU token ID in decoded token list
  4. `infer_streaming()`: Use CacheAwareStreamingAudioBuffer, process chunks, emit partial results via callback. Accumulate final results.
- **Edge cases**: No speakers detected → return empty list. Single speaker → skip diarization overhead. Post-processor not provided → skip refinement, text_refined=None.
- **Reference**: Existing `inference/streaming.py` for CacheAwareStreamingAudioBuffer usage

#### Task 2b: MultitaskEvaluator
- **File**: `src/multitalker_asr/eval/multitask_evaluator.py` — new
- **touches**: [eval/multitask_evaluator.py, eval/__init__.py]
- **provides**: [MultitaskEvaluator]
- **requires**: [MultitaskInferenceEngine]
- **depends_on**: [task-2a]
- **Logic**:
  1. `evaluate()`: Read manifest → for each entry: infer → compare predictions vs ground truth
  2. `_compute_asr_metrics()`: WER and CER using standard edit distance
  3. `_compute_task_accuracy()`: Per-task accuracy (correct/total)
  4. `_generate_report()`: Write markdown report with per-task breakdown
- **Edge cases**: Manifest has no ground truth for some tasks → skip those metrics. Empty predictions → WER=100%, accuracy=0%.

### Wave 3 (CLI + exports + tests)

#### Task 3a: Inference CLI script
- **File**: `scripts/infer_multitask.py` — new
- **touches**: [scripts/infer_multitask.py]
- **provides**: [CLI for multi-task inference]
- **depends_on**: [task-2a]
- **Logic**: argparse: `--audio_path`, `--model_path`, `--use_post_processor`, `--output_json`. Load model → MultitaskInferenceEngine → infer → print/save results.

#### Task 3b: Update exports
- **File**: `src/multitalker_asr/inference/__init__.py` — edit
- **File**: `src/multitalker_asr/eval/__init__.py` — edit
- **File**: `src/multitalker_asr/__init__.py` — edit
- **depends_on**: [task-2a, task-2b]

#### Task 3c: Tests
- **File**: `tests/test_multitask_inference.py` — new
- **touches**: [tests/test_multitask_inference.py]
- **depends_on**: [task-2a, task-2b]
- **Tests**:
  - SpeakerResult: to_dict(), str()
  - _decode_prompt_labels: correct mapping from logits to string labels
  - _detect_eou: finds EOU token, returns False when absent
  - MultitaskEvaluator: metrics computation with mock predictions
  - MultitaskInferenceEngine: infer returns List[SpeakerResult]

## Failure Scenarios

| When | Then | Error |
|------|------|-------|
| No speakers found by Sortformer | Empty result list | Return [], log info |
| RNNT decode produces empty text | SpeakerResult with empty text | Set text="", log warning |
| FunASR post-processor crashes | Refinement skipped | Catch, set text_refined=None, log error |
| Audio file too short (<160ms) | Encoder fails | Check duration >= 160ms, skip if too short |
| Prompt classifier not loaded (Phase 1/2 checkpoint) | Can't decode labels | Return None for all task labels, log warning |

## Rejection Criteria
- DO NOT call FunASR post-processor in the streaming loop — offline only
- DO NOT block on post-processing — make it optional via flag
- DO NOT compute metrics that have no ground truth — skip gracefully
- DO NOT duplicate streaming buffer logic — reuse from existing inference/streaming.py

## Cross-Phase Context
- **Assumes from Phase 1**: TaskTokenRegistry with reverse lookup (int→string)
- **Assumes from Phase 2**: MultitalkerMultiTaskModel with forward_multitask() and prompt_classifier
- **Assumes from Phase 5**: FunASRPostProcessor, VocabularyExtender (EOU token in vocab)
- **Assumes**: Existing inference/streaming.py for CacheAwareStreamingAudioBuffer pattern
- **This is the final phase**: All prior phases must be complete and tested

## Acceptance Criteria
- `uv run pytest tests/test_multitask_inference.py` passes
- `scripts/infer_multitask.py --audio_path demo.wav` produces SpeakerResult output
- Evaluation report includes WER + per-task accuracy
- Streaming inference emits partial results via callback
- System works end-to-end: audio → diarize → transcribe → classify → (optional) refine → output
