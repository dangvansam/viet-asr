# Scout Report: FunASR Multi-task Learning Architecture Analysis

**Scan Date**: 2025-04-21
**Repository**: /home/samdv/multitalker-asr/tmp/FunASR
**Focus**: Fun-ASR-MLT-Nano Model Architecture with Multi-task Learning

---

## EXECUTIVE SUMMARY

FunASR implements a multi-task learning system for ASR through:
1. **Prompt-based task specification** (not token embeddings)
2. **Embedding-based task query injection** at encoder input (prepended)
3. **CTC decoder** with special vocabulary tokens for auxiliary tasks
4. **LLM backbone** (for Nano variant) for flexible task handling
5. **No explicit multi-task training infrastructure** - tasks unified through unified vocabulary

---

## 1. CORE ARCHITECTURE

### SenseVoice Model (Multi-task Base)
**File**: `/home/samdv/multitalker-asr/tmp/FunASR/funasr/models/sense_voice/model.py`

#### Multi-Task Learning Approach:
The model handles multiple tasks through a **single unified CTC output space** that includes special tokens for each task:

```python
# Lines 642-655
self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
self.textnorm_dict = {"withitn": 14, "woitn": 15}

# Embedding layer with 7 + len(lid_dict) + len(textnorm_dict) = 24 special tokens
self.embed = torch.nn.Embedding(24, input_size)

self.emo_dict = {
    "unk": 25009,
    "happy": 25001,
    "sad": 25002,
    "angry": 25003,
    "neutral": 25004,
}
```

**Tasks Supported** (via special tokens):
1. **Language ID (LID)**: 8 special tokens for language selection (Chinese, English, Cantonese, Japanese, Korean, etc.)
2. **Emotion**: 4 tokens embedded in vocabulary (25001-25004) for emotion classification
3. **Text Normalization (ITN)**: 2 tokens for "withitn" vs "woitn" (with/without ITN)

### Fun-ASR-Nano Model (LLM-based Multi-task)
**File**: `/home/samdv/multitalker-asr/tmp/FunASR/funasr/models/fun_asr_nano/model.py`

An **LLM-augmented ASR system** that adds:
- Prompt-based task control (natural language task specification)
- CTC auxiliary decoder
- LLM backbone for arbitrary output generation

```python
# Lines 28-39
class FunASRNano(nn.Module):
    def __init__(self,
        audio_encoder: str = None,
        audio_encoder_conf: dict = None,
        audio_adaptor: str = None,
        llm: str = None,
        llm_conf: dict = None,
        ...
    ):
```

**Component Stack**:
1. `audio_encoder` - Frozen encoder (e.g., Conformer-based from ModelScope)
2. `audio_adaptor` - Projects encoder output to LLM dimension
3. `llm` - Frozen LLM backbone (HuggingFace CausalLM)
4. `ctc_decoder` - Optional CTC head for auxiliary training

---

## 2. PROMPT-BASED TASK INJECTION

### Task Specification via Natural Language Prompts

**File**: `/home/samdv/multitalker-asr/tmp/FunASR/funasr/models/fun_asr_nano/model.py` Lines 550-563

```python
def get_prompt(self, hotwords: list[str], language: str = None, itn: bool = True):
    if len(hotwords) > 0:
        hotwords = ", ".join(hotwords)
        prompt = f"请结合上下文信息，更加准确地完成语音转写任务。..."
        prompt += f"热词列表：[{hotwords}]\n"
    else:
        prompt = ""
    
    if language is None:
        prompt += "语音转写"
    else:
        prompt += f"语音转写成{language}"
    
    if not itn:
        prompt += "，不进行文本规整"
    return prompt + "："
```

**Prompt Parameters (Task Controls)**:
- `language`: 31 languages (Chinese, English, Japanese, Korean, Vietnamese, Thai, etc.)
- `itn`: Boolean flag for Inverse Text Normalization
- `hotwords`: Context-specific vocabulary words

### Chat Format with Audio Injection

**File**: Lines 565-581

```python
def generate_chatml(self, prompt: str, data: Union[str, torch.Tensor]):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user", 
            "content": f"{prompt}<|startofspeech|>!{data}<|endofspeech|>",
            "audio": data,  # Audio tensor injected here
        },
        {"role": "assistant", "content": "null"},
    ]
```

**Key Pattern**:
- Task prompt + `<|startofspeech|>` + audio_reference + `<|endofspeech|>`
- Audio frames later replaced via `fbank_beg` indexing

---

## 3. ENCODER-SIDE TASK QUERY INJECTION

### SenseVoice Approach (Embedding-based Prepending)

**File**: Lines 722-774 (encode method)

```python
def encode(self, speech, speech_lengths, text):
    # ... feature extraction ...
    
    # Language ID query embedding (1 token)
    lids = torch.LongTensor([
        [self.lid_int_dict[int(lid)] if torch.rand(1) > 0.2 
         else 0]
        for lid in text[:, 0]
    ]).to(speech.device)
    language_query = self.embed(lids)  # [batch, 1, input_size]
    
    # Text normalization query (1 token)
    styles = torch.LongTensor(
        [[self.textnorm_int_dict[int(style)]] for style in text[:, 3]]
    ).to(speech.device)
    style_query = self.embed(styles)
    speech = torch.cat((style_query, speech), dim=1)
    speech_lengths += 1
    
    # Event + Emotion queries (2 tokens)
    event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(
        speech.size(0), 1, 1
    )
    
    # Combine all task queries + speech
    input_query = torch.cat((language_query, event_emo_query), dim=1)
    speech = torch.cat((input_query, speech), dim=1)
    speech_lengths += 3  # 3 prepended tokens total
    
    encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
    return encoder_out, encoder_out_lens
```

**Task Query Structure**:
```
[LID_token] + [Event_token, Emotion_token] + [ITN_token] + [speech_frames...]
     1            2                              1            T frames
     └─── 4 prepended special tokens before encoder ────┘
```

**Inference Mode** (Lines 852-876):

```python
# Language control via parameter
language = kwargs.get("language", "auto")
language_query = self.embed(
    torch.LongTensor([[self.lid_dict[language]...]]).to(speech.device)
)

# ITN control via parameter
textnorm = kwargs.get("text_norm", None)  # "withitn" or "woitn"
if textnorm is None:
    textnorm = "withitn" if use_itn else "woitn"
textnorm_query = self.embed(
    torch.LongTensor([[self.textnorm_dict[textnorm]]]).to(speech.device)
)

# Prepend to speech frames
speech = torch.cat((textnorm_query, speech), dim=1)
speech_lengths += 1
```

---

## 4. CTC DECODER & SPECIAL TOKENS

### CTC Module Architecture

**File**: `/home/samdv/multitalker-asr/tmp/FunASR/funasr/models/fun_asr_nano/ctc.py`

```python
class CTC(torch.nn.Module):
    def __init__(self, odim: int, encoder_output_size: int, 
                 dropout_rate: float = 0.0, blank_id: int = 0, **kwargs):
        super().__init__()
        self.ctc_lo = torch.nn.Linear(eprojs, odim)  # Linear projection to vocab
        self.blank_id = blank_id
        self.ctc_loss = torch.nn.CTCLoss(reduction="none", blank=blank_id)
```

**Vocabulary Space**:
- Token range: [0, vocab_size) typically 60515 for multilingual
- Special tokens for auxiliary tasks:
  - `25001-25004`: Emotion tokens (happy, sad, angry, neutral)
  - `25009`: Unknown/unspecified emotion
  - Task tokens embedded at encoder input (not in CTC output vocab)

### Loss Computation

**File**: SenseVoice `_calc_ctc_loss` (Lines 776-790):

```python
def _calc_ctc_loss(self, encoder_out, encoder_out_lens, ys_pad, ys_pad_lens):
    loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
    return loss_ctc, cer_ctc
```

**Loss Combination** (Lines 709):

```python
# SenseVoice combines two losses:
loss_ctc, cer_ctc = self._calc_ctc_loss(
    encoder_out[:, 4:, :],      # Skip 4 task tokens
    encoder_out_lens - 4,
    text[:, 4:],                # Skip task token targets
    text_lengths - 4
)
loss_rich, acc_rich = self._calc_rich_ce_loss(
    encoder_out[:, :4, :],      # Only task token frames
    text[:, :4]                 # Task token targets
)
loss = loss_ctc + loss_rich
```

**Two-Stage Training**:
1. **Auxiliary head loss** (`loss_rich`): Learn task embeddings (emotion, language, ITN)
2. **Main CTC loss** (`loss_ctc`): Learn ASR transcription

---

## 5. FUN-ASR-NANO: LLM-BASED MULTI-TASK UNIFICATION

### Data Template and Prompt Integration

**File**: `/home/samdv/multitalker-asr/tmp/FunASR/funasr/models/fun_asr_nano/model.py` Lines 308-459

**Data Loading Flow**:

```python
def data_load_speech(self, contents: dict, tokenizer, frontend, **kwargs):
    system = contents["system"]
    user = contents["user"]
    assistant = contents["assistant"]
    
    # Pattern for locating audio
    pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
    
    for i, (system_prompt, user_prompt, target_out) in enumerate(
        zip(system, user, assistant)
    ):
        splits = pattern.split(source_input)
        
        for k, sub_str in enumerate(splits):
            if sub_str.startswith("<|startofspeech|>"):
                # Audio token placeholder
                fake_token = [0] * fake_token_len_i
                fbank_beg_i = len(source_ids)
                source_ids += fake_token  # Placeholder for audio
            else:
                # Text tokens from prompt
                sub_token = tokenizer.encode(sub_str)
                source_ids += sub_token
```

**Audio-Text Embedding Injection** (Lines 161-227, `forward` method):

```python
# Get input embeddings for all tokens (text + placeholder)
inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)

# Replace placeholder tokens with actual audio embeddings
for batch_idx in range(batch_size):
    for turn_id in range(fbank_beg.shape[1]):
        fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
        if fbank_beg_idx > 0:
            speech_token_len = fake_token_len[batch_idx, turn_id]
            speech_token = encoder_out[speech_idx, :speech_token_len, :]
            
            inputs_embeds[
                batch_idx,
                fbank_beg_idx : fbank_beg_idx + speech_token_len,
                :,
            ] = speech_token
            
            speech_idx += 1
```

**Key Difference from SenseVoice**:
- NO prepended task query tokens
- Task control via LLM prompt tokens (natural language)
- Audio embeddings replace placeholder token positions
- LLM generates flexible outputs (not constrained to vocab)

---

## 6. MULTI-TASK LOSS FUNCTIONS

### Label Smoothing Loss

**File**: `/home/samdv/multitalker-asr/tmp/FunASR/funasr/losses/label_smoothing_loss.py`

```python
class LabelSmoothingLoss(nn.Module):
    def forward(self, x, target):
        # x: [batch, seqlen, class]
        # target: [batch, seqlen]
        x = x.contiguous().view(-1, self.size)
        target = target.contiguous().view(-1)
        
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target == self.padding_idx
            target = target.masked_fill(ignore, 0)
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom
```

### Training Statistics Tracking

Both models track:
- `loss`: Main task loss
- `loss_ctc`: CTC component (SenseVoice only)
- `loss_rich`: Auxiliary task loss (SenseVoice only)
- `acc`: Token accuracy
- `acc_rich`: Auxiliary task accuracy

---

## 7. TASK-SPECIFIC INFERENCE

### SenseVoice Inference (Controlled via Vocab Tokens)

**File**: Lines 809-950

```python
# Set language and ITN via embedding lookup
language = kwargs.get("language", "auto")  # auto, zh, en, yue, ja, ko
language_query = self.embed(
    torch.LongTensor([[self.lid_dict[language]]]).to(speech.device)
)

# Suppress unwanted emotion tokens in output
if kwargs.get("ban_emo_unk", False):
    ctc_logits[:, :, self.emo_dict["unk"]] = -float("inf")

# Decode and extract special token predictions
yseq = x.argmax(dim=-1)  # Greedy CTC decode
mask = yseq != self.blank_id
token_int = yseq[mask].tolist()
text = tokenizer.decode(token_int)
```

### Fun-ASR-Nano Inference (Controlled via Prompt)

**File**: Lines 583-738

```python
def inference(self, data_in, tokenizer, frontend, **kwargs):
    prompt = self.get_prompt(
        kwargs.get("hotwords", []),           # Context words
        kwargs.get("language", None),         # Target language
        kwargs.get("itn", True)               # ITN flag
    )
    
    # Integrate prompt into chat format
    data_in = [self.generate_chatml(prompt, data) for data in data_in]
    
    # LLM generates output with prompt guidance
    generated_ids = self.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=kwargs.get("max_length", 512),
        **llm_kwargs,
    )
```

**Supported Tasks via Prompt**:
1. Language selection (31 languages)
2. Hotword/context incorporation
3. ITN on/off
4. Multi-turn dialogue (up to 5 turns)

---

## 8. DIRECTORY STRUCTURE

```
/home/samdv/multitalker-asr/tmp/FunASR/
├── funasr/
│   ├── models/
│   │   ├── sense_voice/
│   │   │   ├── model.py              [SenseVoice encoder + CTC + special tokens]
│   │   │   ├── utils/
│   │   │   └── whisper_lib/          [Tokenizer, language detection]
│   │   ├── fun_asr_nano/
│   │   │   ├── model.py              [FunASR-Nano: LLM + audio adaptor + CTC]
│   │   │   ├── ctc.py                [CTC loss module]
│   │   │   └── tools/utils.py        [Forced alignment]
│   │   ├── paraformer/               [Base ASR encoder]
│   │   ├── ctc/ctc.py                [Generic CTC implementation]
│   │   └── ... (45 model variants)
│   ├── losses/
│   │   ├── label_smoothing_loss.py   [Loss with smoothing]
│   │   └── ... (other loss functions)
│   └── utils/
│       ├── load_utils.py             [Audio/video/image loading]
│       └── datadir_writer.py         [Output writing]
├── examples/industrial_data_pretraining/
│   ├── sense_voice/                  [SenseVoice training examples]
│   └── fun_asr_nano/                 [FunASR-Nano training examples]
└── runtime/                          [ONNX/TorchScript deployment]
```

---

## 9. KEY IMPLEMENTATION DETAILS

### Embedding Dimensions
```python
# SenseVoice
embed = nn.Embedding(7 + 8 + 2, input_size=80)  # 17 special tokens → 80-dim embeddings
# Tasks: 7 base + 8 language IDs + 2 ITN flags
```

### Token Count Adjustments
```python
# Speech lengths increase by prepended task tokens
speech_lengths += 1      # ITN token
speech_lengths += 3      # Language + Event + Emotion tokens
# Total: 4 extra tokens before speech frames in SenseVoice
```

### CTC Blank Handling
```python
self.blank_id = blank_id or vocab_size - 1
self.ctc_loss = torch.nn.CTCLoss(reduction="none", blank=blank_id)
```

---

## 10. VOCABULARY STRUCTURE

### SenseVoice Vocabulary (Pinyin-based Chinese + multilingual)
- **0-23**: Task special tokens (LID, emotion, ITN)
- **24-25000**: Regular characters/phonemes
- **25001-25004**: Emotion labels (happy, sad, angry, neutral)
- **25009**: Unknown emotion
- **25016-25017**: ITN control tokens

### Fun-ASR-Nano Vocabulary
- Inherited from base LLM (e.g., Qwen)
- No hard-coded emotion/language tokens
- Task control via text prompt

---

## 11. INTEGRATION POINTS FOR MULTITALKER-ASR

### Applicable Patterns:
1. **Task query prepending** (SenseVoice): Embed speaker IDs/diarization states at input
2. **CTC auxiliary head** (both): Add speaker prediction head
3. **Prompt control** (Fun-ASR-Nano): "Recognize speech from Speaker A"
4. **Multi-stage loss** (SenseVoice): Combine ASR + speaker classification loss
5. **Vocabulary extension**: Add speaker tokens (e.g., `<speaker_1>`, `<speaker_2>`)

### NOT Applicable:
- SenseVoice emotion learning ❌ (no emotion task in multitalker-asr)
- LLM prompt flexibility ✓ (could specify speaker context)

---

## 12. FILE INVENTORY (Key Files Only)

| File | Lines | Purpose |
|------|-------|---------|
| `funasr/models/sense_voice/model.py` | 1000+ | Main SenseVoice model with task embeddings |
| `funasr/models/fun_asr_nano/model.py` | 750+ | Fun-ASR-Nano LLM-based model |
| `funasr/models/fun_asr_nano/ctc.py` | 61 | CTC loss and decoding |
| `funasr/losses/label_smoothing_loss.py` | 124 | Training loss with smoothing |
| `funasr/models/paraformer/` | 400+ | Base encoder architecture |
| `examples/industrial_data_pretraining/fun_asr_nano/model.py` | 750+ | Example training code |
| `examples/industrial_data_pretraining/sense_voice/` | 500+ | SenseVoice examples |

**Total Model Files**: 276 Python files across all model variants

---

## SUMMARY TABLE

| Feature | SenseVoice | Fun-ASR-Nano |
|---------|-----------|--------------|
| **Base Architecture** | Conformer encoder + CTC | Frozen encoder + LLM |
| **Task Control** | Embedding tokens prepended | Natural language prompt |
| **Vocabulary** | Fixed (special tokens 0-25017) | LLM vocab (inherited) |
| **Tasks Supported** | Language, Emotion, ITN | Language, ITN, Hotwords, Multi-turn |
| **Auxiliary Loss** | Emotion prediction head | CTC auxiliary (optional) |
| **Inference** | Greedy CTC decode | LLM generation |
| **Training** | Multi-task joint training | LLM + CTC co-training |
| **Flexibility** | Limited to predefined tasks | Arbitrary text generation |

