# FunASR Integration Guide for Multitalker ASR

**Date**: 2025-04-21
**Source**: Complete scan of `/home/samdv/multitalker-asr/tmp/FunASR`
**Status**: Ready for implementation

---

## DOCUMENTS IN THIS GUIDE

This guide consists of 3 complementary documents:

### 1. **FUNASR_ARCHITECTURE_ANALYSIS.md** (531 lines, 18K)
Comprehensive technical deep-dive covering:
- Executive summary of FunASR's multi-task approach
- SenseVoice model architecture (emotion + language + ITN)
- Fun-ASR-Nano LLM-based architecture
- Complete prompt-based task injection mechanism
- CTC decoder with special tokens
- Loss functions and training strategy
- Full directory structure and file inventory
- Vocabulary organization (0-25017 range)
- Integration points for multitalker-asr
- Comparison table: SenseVoice vs Fun-ASR-Nano

**When to use**: Understanding the full architecture before implementation

---

### 2. **FUNASR_ARCHITECTURE_QUICK_REFERENCE.md** (247 lines, 7.9K)
Practical quick-reference for implementation:
- Two distinct multi-task approaches (SenseVoice vs Fun-ASR-Nano)
- Exact task token prepending pattern
- Audio-text embedding injection mechanism
- Loss computation (two-stage training)
- Key differences table
- Integration templates for multitalker-asr
- Critical implementation details
- Vocabulary extension patterns
- Design recommendations

**When to use**: During active implementation, checking specific patterns

---

### 3. **FUNASR_CODE_SNIPPETS.md** (589 lines, 19K)
Copy-paste ready code implementations:
1. Speaker embedding lookup table
2. Task query prepending in encode()
3. Two-stage loss computation
4. CrossEntropyLoss for speaker classification
5. Inference with speaker control
6. Audio-text embedding injection (alternative approach)
7. Vocabulary extension configuration
8. Minimal CTC module
9. Label smoothing loss
10. Complete working example (SimpleMultitalkerASR)
11. Implementation checklist

**When to use**: Actual coding - copy patterns directly into your codebase

---

## QUICK START: 3-STEP INTEGRATION

### Step 1: Choose Your Approach
Read **FUNASR_ARCHITECTURE_QUICK_REFERENCE.md** Section "THE BOTTOM LINE"

**Recommendation for multitalker-asr**: **Option A (SenseVoice-like)**
- Why: Fixed speaker count, deterministic, lightweight, interpretable
- Pattern: Prepend speaker task tokens before encoder

### Step 2: Understand the Pattern
Read **FUNASR_ARCHITECTURE_QUICK_REFERENCE.md** Sections:
1. "APPROACH 1: SenseVoice (Embedding-Token Prepending)"
2. "CRITICAL IMPLEMENTATION DETAILS"
3. "INTEGRATION TEMPLATE FOR MULTITALKER-ASR"

### Step 3: Implement
Use **FUNASR_CODE_SNIPPETS.md**:
1. Copy Snippet #1: Speaker embedding table
2. Copy Snippet #2: Task prepending in encode()
3. Copy Snippet #3: Two-stage loss
4. Copy Snippet #8: CTC module
5. Use Snippet #10: Complete example as template
6. Follow "IMPLEMENTATION CHECKLIST"

---

## THE EXACT ARCHITECTURE PATTERN (FOR SPEAKERS)

```
┌─────────────────────────────────────────────────────────┐
│ Input: audio_frames [batch, T, 80]                      │
│        speaker_labels [batch]                           │
└────────────────┬────────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────────────┐
│ 1. EMBEDDING (before encoder)                           │
│    speaker_query = speaker_embed[speaker_labels]        │
│    → [batch, 1, 256]                                    │
└────────────────┬────────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────────────┐
│ 2. PREPEND to speech                                    │
│    speech_with_task = cat([speaker_query, speech])      │
│    → [batch, T+1, 80]                                   │
│    speech_lengths += 1                                  │
└────────────────┬────────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────────────┐
│ 3. ENCODER (Conformer/Paraformer/etc)                   │
│    encoder_out, encoder_out_lens = self.encoder(...)    │
│    → encoder_out [batch, T+1, 256]                      │
└────────────────┬────────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────────────┐
│ 4. LOSS COMPUTATION (Two stages)                        │
│                                                          │
│    Stage 1 - Speaker Classification:                    │
│    speaker_logits = linear(encoder_out[:, 0, :])        │
│    loss_speaker = CE(speaker_logits, speaker_labels)    │
│                                                          │
│    Stage 2 - ASR (CTC):                                 │
│    asr_logits = ctc_linear(encoder_out[:, 1:, :])       │
│    loss_ctc = CTC(asr_logits, text_targets)             │
│                                                          │
│    Combined:                                             │
│    loss = loss_ctc + 0.1 * loss_speaker                 │
└────────────────┬────────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────────────┐
│ 5. INFERENCE                                            │
│    Given: speaker_id, audio                             │
│    → Text output + (optional) speaker_confidence        │
└─────────────────────────────────────────────────────────┘
```

---

## KEY VOCABULARY CONFIGURATION

```python
# Vocabulary split (adapt to your tokenizer)
BLANK_ID = 0
PAD_ID = 1
ASR_VOCAB_START = 2
ASR_VOCAB_END = 60000      # Characters/phonemes

# Additional tokens for speakers (optional)
SPEAKER_TOKENS_START = 60001
# speaker_0: 60001
# speaker_1: 60002
# etc.
```

---

## CRITICAL SUCCESS FACTORS

1. **Token counting**: Always adjust `speech_lengths += num_prepended_tokens`
2. **Loss slicing**: First K encoder outputs for task loss, rest for CTC
3. **Batch consistency**: All samples must have same number of task tokens
4. **Inference mode**: Specify speaker_id at test time for deterministic results
5. **Weight balancing**: Auxiliary loss weight (typically 0.01-0.1)

---

## TROUBLESHOOTING

| Issue | Solution |
|-------|----------|
| Shape mismatch in loss | Check `encoder_out` slicing - offset must match prepended tokens |
| NaN in training | Reduce auxiliary loss weight or enable label smoothing |
| Speaker prediction always same | Check embedding initialization, may need label smoothing |
| Inference failing | Verify speaker_id is in valid range [0, num_speakers) |
| Different results each run | Set random seeds, freeze encoder if needed |

---

## FILE REFERENCES FROM FUNASR

**Original source files** (read-only reference):
- `/home/samdv/multitalker-asr/tmp/FunASR/funasr/models/sense_voice/model.py`
  - Lines 642-648: Task embedding table
  - Lines 722-774: Task prepending in encode()
  - Lines 707-720: Two-stage loss computation
  - Lines 852-876: Inference with task control

- `/home/samdv/multitalker-asr/tmp/FunASR/funasr/models/fun_asr_nano/model.py`
  - Lines 161-227: Audio embedding injection (alternative)
  - Lines 550-563: Prompt-based task specification
  - Lines 565-581: Chat format with audio

- `/home/samdv/multitalker-asr/tmp/FunASR/funasr/models/fun_asr_nano/ctc.py`
  - Complete CTC loss implementation

- `/home/samdv/multitalker-asr/tmp/FunASR/funasr/losses/label_smoothing_loss.py`
  - Label smoothing for smoother training

---

## NEXT STEPS

1. **Phase 1**: Read all 3 documents in order
2. **Phase 2**: Sketch integration plan using QUICK_REFERENCE
3. **Phase 3**: Implement using CODE_SNIPPETS
4. **Phase 4**: Test with dummy data (see Snippet #10 example)
5. **Phase 5**: Integrate with real training pipeline
6. **Phase 6**: Evaluate on multitalker test set

---

## DOCUMENT MATRIX

| Question | Document | Section |
|----------|----------|---------|
| How does multi-task learning work in FunASR? | ANALYSIS | Section 1-5 |
| What are the two approaches? | QUICK_REF | "APPROACH 1 & 2" |
| How do I prepend speaker tokens? | QUICK_REF | "CRITICAL DETAILS" |
| Show me speaker embedding code | CODE_SNIPPETS | Snippet #1 |
| How is loss computed? | CODE_SNIPPETS | Snippet #3 |
| What's the complete flow? | CODE_SNIPPETS | Snippet #10 |
| What should I implement first? | QUICK_REF | "3-STEP INTEGRATION" |
| What vocabulary changes are needed? | CODE_SNIPPETS | Snippet #7 |

---

## CONTACT REFERENCES

All information extracted from:
- FunASR GitHub: https://github.com/FunAudioLLM/Fun-ASR
- ModelScope: FunAudioLLM/Fun-ASR-MLT-Nano-2512
- HuggingFace: FunAudioLLM/Fun-ASR-MLT-Nano-2512

---

**Generated**: 2025-04-21
**Version**: 1.0
**Status**: Ready for implementation

