# Phase 5: EOU + Fun-ASR-MLT-Nano ITN/PnC Post-Processor

## Data Flow
```
RNNT Vocabulary Extension:
  existing_vocab: [token_0, ..., token_2047]
    + <EOU> token → token_2048
    → EncDecMultiTalkerRNNTBPEModel.change_vocabulary()

Streaming ASR Output (per speaker):
  "xin chao ban khoe khong<EOU>"
       ↓
FunASRPostProcessor.refine(raw_text, language="vi", itn=True)
       ↓
  Fun-ASR-MLT-Nano model.generate(input=raw_audio, itn=True)
       ↓
  "Xin chào, bạn khỏe không?"  (ITN + PnC applied)
```

## Code Contracts

```python
# --- models/vocab_extension.py ---
class VocabularyExtender:
    """Extends RNNT vocabulary with special tokens (<EOU>, etc.)."""
    SPECIAL_TOKENS = {"<EOU>": "end_of_utterance"}

    @staticmethod
    def extend_rnnt_vocab(
        model: EncDecMultiTalkerRNNTBPEModel,
        special_tokens: List[str],
        tokenizer_dir: str,
    ) -> int:
        """Add special tokens to tokenizer + resize model embeddings.
        Returns new vocab size."""

    @staticmethod
    def create_extended_tokenizer(
        base_tokenizer_dir: str,
        output_dir: str,
        special_tokens: List[str],
    ) -> str:
        """Create new tokenizer with added special tokens. Returns output_dir."""

# --- inference/post_processor.py ---
class FunASRPostProcessor:
    """Wraps Fun-ASR-MLT-Nano for ITN + PnC post-processing."""
    def __init__(
        self,
        model_name: str = "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
        device: str = "cuda:0",
    ): ...

    def load(self) -> None:
        """Lazy-load the FunASR model."""

    def refine_text(
        self,
        raw_text: str,
        audio_path: Optional[str] = None,
        language: str = "auto",
        itn: bool = True,
    ) -> str:
        """Apply ITN + PnC to raw ASR output.
        If audio_path provided: re-transcribe with FunASR (higher quality).
        If only raw_text: apply text-level ITN rules only."""

    def refine_batch(
        self,
        items: List[Dict[str, Any]],  # [{"text": str, "audio_path": str}, ...]
        language: str = "auto",
        itn: bool = True,
    ) -> List[Dict[str, Any]]:
        """Batch refinement. Returns items with added 'refined_text' key."""
```

## Tasks

### Wave 1 (independent)

#### Task 1a: VocabularyExtender
- **File**: `src/multitalker_asr/models/vocab_extension.py` — new
- **touches**: [models/vocab_extension.py, models/__init__.py]
- **provides**: [VocabularyExtender]
- **requires**: []
- **Logic**:
  1. `create_extended_tokenizer()`: Copy base tokenizer dir, read vocab file, append special tokens with new IDs, write updated vocab. Handle SentencePiece `.model` files by creating a user-defined symbols file.
  2. `extend_rnnt_vocab()`: Call `model.change_vocabulary(new_tokenizer_dir)` which resizes embedding + joint layers. Log old/new vocab size.
- **Edge cases**: Token already exists in vocab → skip, log info. Tokenizer dir doesn't exist → FileNotFoundError. SentencePiece model is binary → must use NeMo's change_vocabulary API, not raw file edit.
- **Reference**: Existing `models/tokenizer_extender.py` — check if this already handles some of this.

#### Task 1b: FunASRPostProcessor
- **File**: `src/multitalker_asr/inference/post_processor.py` — new
- **touches**: [inference/post_processor.py, inference/__init__.py]
- **provides**: [FunASRPostProcessor]
- **requires**: []
- **Logic**:
  1. `__init__()`: Store model_name, device. Don't load model yet (lazy).
  2. `load()`: `from funasr import AutoModel; self._model = AutoModel(model=model_name, trust_remote_code=True, device=device)`. Following test_asr_punctuation.py pattern exactly.
  3. `refine_text()`: If audio_path provided → `self._model.generate(input=[audio_path], cache={}, language=language, itn=itn)` → return result[0]["text"]. If only raw_text → return raw_text as-is (text-only ITN not supported by this model).
  4. `refine_batch()`: Loop over items, call refine_text per item.
- **Edge cases**: Model not loaded → auto-call load(). FunASR not installed → ImportError with helpful message. GPU OOM → catch, retry on CPU.
- **IMPORTANT**: The FunASR model requires audio input for ITN/PnC — it re-transcribes. This means post-processing needs original audio, not just text.

### Wave 2 (exports + tests)

#### Task 2a: Update exports
- **File**: `src/multitalker_asr/models/__init__.py` — edit
- **File**: `src/multitalker_asr/inference/__init__.py` — edit
- **File**: `src/multitalker_asr/__init__.py` — edit
- **depends_on**: [task-1a, task-1b]

#### Task 2b: Unit tests
- **File**: `tests/test_vocab_extension.py` — new
- **File**: `tests/test_post_processor.py` — new
- **touches**: [tests/]
- **depends_on**: [task-1a, task-1b]
- **Tests**:
  - VocabularyExtender: create_extended_tokenizer adds tokens to vocab file
  - VocabularyExtender: duplicate token is skipped
  - FunASRPostProcessor: instantiation without loading model
  - FunASRPostProcessor: refine_text with mock model returns expected output
  - FunASRPostProcessor: auto-load on first refine_text call

## Failure Scenarios

| When | Then | Error |
|------|------|-------|
| FunASR package not installed | ImportError | Catch, raise with "pip install funasr" instruction |
| FunASR model download fails (network) | Model load timeout | Retry once, then raise with manual download instructions |
| Audio path doesn't exist for refine_text | FunASR crashes | Validate path exists before calling generate |
| SentencePiece model can't be extended directly | Vocab extension fails | Use NeMo's change_vocabulary() which handles this |

## Rejection Criteria
- DO NOT modify SentencePiece .model binary directly — use NeMo's change_vocabulary API
- DO NOT make FunASR a hard dependency — it's optional for post-processing only
- DO NOT call FunASR in the streaming loop — it's offline post-processing only
- DO NOT remove existing TokenizerExtender — VocabularyExtender complements it

## Cross-Phase Context
- **Assumes**: TokenizerExtender exists at `models/tokenizer_extender.py` — may overlap, check
- **Assumes from Phase 2**: MultitalkerMultiTaskModel has accessible RNNT decoder
- **Exports for Phase 6**: VocabularyExtender (called during model init), FunASRPostProcessor (called in inference pipeline)
- **Note**: FunASR post-processing requires original audio — Phase 6 inference must pass audio_path alongside RNNT output

## Acceptance Criteria
- `uv run pytest tests/test_vocab_extension.py tests/test_post_processor.py` passes
- VocabularyExtender can add <EOU> token without breaking existing tokenizer
- FunASRPostProcessor.refine_text() with audio produces ITN+PnC output
- Post-processor is optional — system works without it (raw RNNT output)
