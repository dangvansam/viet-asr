# FunASR Multi-Task Architecture - Quick Reference for Integration

## THE EXACT IMPLEMENTATION PATTERN

FunASR uses **TWO DISTINCT APPROACHES** for multi-task learning:

---

## APPROACH 1: SenseVoice (Embedding-Token Prepending)

### Architecture:
```
Input: [speech_frames]
  ↓
Extract fbank features
  ↓
ENCODE PHASE (before encoder):
  - Create task query embeddings from LookUp Table
  - PREPEND to speech: [task_token_1, task_token_2, ..., task_token_N] + [speech_frames]
  ↓
Pass through Conformer encoder
  ↓
Output CTC logits (covering both ASR + task tokens in vocabulary)
  ↓
Loss: CTCLoss + CrossEntropyLoss on task positions
```

### Task Token Structure (SenseVoice):
```python
# Line 642-648 of funasr/models/sense_voice/model.py
self.embed = torch.nn.Embedding(24, input_size=80)  # 24 special tokens → 80-dim embeddings

Task queries (4 tokens prepended):
Position 0: Language ID token     [1 token] → embedding[lid_dict[lang]]
Position 1-2: Event + Emotion     [2 tokens] → embedding[[1, 2]]  
Position 3: Text Normalization    [1 token] → embedding[textnorm_dict['withitn'/'woitn']]
Position 4+: Speech frames        [T tokens]
```

### Loss Computation (Two Stages):
```python
# Line 707-709
loss_rich = self._calc_rich_ce_loss(encoder_out[:, :4, :], text[:, :4])  # Task tokens only
loss_ctc = self._calc_ctc_loss(encoder_out[:, 4:, :], text[:, 4:])       # Speech tokens only
loss = loss_ctc + loss_rich  # Combined loss
```

### Inference:
```python
# Line 852-876
language_query = self.embed(torch.LongTensor([[self.lid_dict[language]]]))
speech = torch.cat((language_query, event_emo_query, textnorm_query, speech), dim=1)
speech_lengths += 3
encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
```

---

## APPROACH 2: Fun-ASR-Nano (LLM + Prompt-Based)

### Architecture:
```
Input: prompt + [<|startofspeech|>] + audio_reference + [<|endofspeech|>]
  ↓
TOKENIZE phase:
  - Tokenize prompt to token IDs
  - Create PLACEHOLDER tokens for audio position
  - Combine: [prompt_tokens] + [0, 0, ..., 0] + [response_tokens]
  ↓
EMBEDDING phase:
  - Get LLM embeddings for all tokens
  - REPLACE placeholder embeddings with audio encoder outputs
  ↓
Forward through frozen LLM
  ↓
Generate output via LLM.generate() (flexible, not constrained to vocab)
```

### Task Control via Prompt:
```python
# Line 550-563
def get_prompt(self, hotwords: list[str], language: str = None, itn: bool = True):
    prompt = "请结合上下文信息..."
    prompt += f"热词列表：[{', '.join(hotwords)}]\n"
    prompt += f"语音转写成{language}"
    if not itn:
        prompt += "，不进行文本规整"
    return prompt + "："
```

### Audio Embedding Injection:
```python
# Line 172-227
inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)

for batch_idx in range(batch_size):
    for turn_id in range(fbank_beg.shape[1]):
        fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
        speech_token_len = fake_token_len[batch_idx, turn_id]
        speech_token = encoder_out[speech_idx, :speech_token_len, :]
        
        # REPLACE placeholders with real audio embeddings
        inputs_embeds[batch_idx, fbank_beg_idx:fbank_beg_idx+speech_token_len, :] = speech_token
```

### Inference:
```python
# Line 664-670
generated_ids = self.llm.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    max_new_tokens=512,
    pad_token_id=self.llm.config.pad_token_id or self.llm.config.eos_token_id,
)
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

---

## KEY DIFFERENCES

| Aspect | SenseVoice | Fun-ASR-Nano |
|--------|-----------|--------------|
| **Task specification** | Prepended embedding tokens | Natural language prompt |
| **Number of tasks** | Fixed (4 token positions) | Arbitrary (via LLM flexibility) |
| **Output space** | Fixed vocabulary (60k tokens) | LLM vocabulary (inherited) |
| **Task control mechanism** | Embedding lookup table | Text prompt |
| **Decoding** | Greedy CTC (argmax) | LLM generation (beam/sampling) |
| **Auxiliary loss** | Cross-entropy on task positions | CTC loss (optional) |
| **Scalability** | Limited to predefined tasks | Unlimited (LLM native) |

---

## INTEGRATION TEMPLATE FOR MULTITALKER-ASR

### Using SenseVoice Pattern (Speaker IDs):

```python
# Create speaker embedding lookup
self.speaker_dict = {
    "speaker_0": 0, "speaker_1": 1, "speaker_2": 2, ...
}
self.speaker_embed = torch.nn.Embedding(num_speakers, input_size=80)

# In encode():
speaker_ids = torch.LongTensor([[speaker_dict[spk_id]]]).to(speech.device)
speaker_query = self.speaker_embed(speaker_ids)

# Prepend like SenseVoice does
speech = torch.cat((speaker_query, speech), dim=1)
speech_lengths += 1

# Loss computation
loss_speaker = self.speaker_loss(encoder_out[:, :1, :], speaker_labels)
loss_asr = self.ctc(encoder_out[:, 1:, :], ...)
loss = loss_asr + 0.1 * loss_speaker
```

### Using Fun-ASR-Nano Pattern (Speaker Context):

```python
def get_prompt(self, speaker_id: str, lang: str = None):
    prompt = f"Speaker {speaker_id}: Transcribe the speech to {lang}:"
    return prompt

# Inference
prompt = self.get_prompt(speaker_id="A", lang="Vietnamese")
data_in = self.generate_chatml(prompt, audio)
# Rest follows Fun-ASR-Nano pattern
```

---

## CRITICAL IMPLEMENTATION DETAILS

### SenseVoice Task Token Prepending:
- **When**: During both training and inference
- **Order**: `[lang_token] + [event_token, emotion_token] + [textnorm_token] + [speech]`
- **Length adjustment**: `speech_lengths += number_of_prepended_tokens`
- **Loss split**: Separate CE loss for first K frames, CTC loss for remaining

### Fun-ASR-Nano Placeholder Replacement:
- **When**: After tokenization, before forward pass
- **What**: Replace token ID placeholders with actual audio embeddings
- **How**: Index into `inputs_embeds` at `fbank_beg` positions
- **Key variables**: 
  - `fbank_beg`: Starting index of audio in token sequence
  - `fake_token_len`: Number of audio frames (as tokens)

---

## VOCABULARY EXTENSION PATTERNS

### SenseVoice (Add speaker tokens to vocab):
```python
# Extend vocabulary to include speaker tokens
vocab_size = 60515 + num_speakers
# Prepend speaker task queries (like emotion tokens 25001-25004)
# Add speaker output tokens to CTC vocabulary

self.speaker_tokens = {
    "speaker_A": 60516,
    "speaker_B": 60517,
    ...
}
```

### Fun-ASR-Nano (No vocab change needed):
```python
# Speaker control entirely via prompt text:
# "Speaker A: Transcribe to Vietnamese"
# LLM handles speaker semantics through language understanding
# No new tokens required
```

---

## THE BOTTOM LINE FOR MULTITALKER-ASR DESIGN

**Option A (SenseVoice-like):**
- Pros: Explicit task control, lightweight, proven
- Cons: Requires vocabulary changes, hardcoded task count
- Best for: Fixed number of speakers, deterministic behavior

**Option B (Fun-ASR-Nano-like):**
- Pros: Flexible, no vocab changes, LLM can reason about speakers
- Cons: Higher latency, more parameters, less interpretable
- Best for: Variable speakers, complex speaker relationships, flexible output

**Recommended for multitalker-asr**: Hybrid approach using **Option A (SenseVoice)** because:
1. Speaker count is typically fixed or bounded
2. Deterministic speaker assignment is critical
3. Lightweight & interpretable is needed for diarization
4. Pattern is well-tested in FunASR already

---

## FILE REFERENCES

| File | Use Case |
|------|----------|
| `/home/samdv/multitalker-asr/tmp/FunASR/funasr/models/sense_voice/model.py:642-774` | Copy task prepending logic |
| `/home/samdv/multitalker-asr/tmp/FunASR/funasr/models/sense_voice/model.py:776-810` | Copy loss computation |
| `/home/samdv/multitalker-asr/tmp/FunASR/funasr/models/fun_asr_nano/model.py:161-227` | Copy embedding injection logic |
| `/home/samdv/multitalker-asr/tmp/FunASR/funasr/models/fun_asr_nano/ctc.py` | Copy CTC module |
| `/home/samdv/multitalker-asr/tmp/FunASR/funasr/losses/label_smoothing_loss.py` | Loss reference |

