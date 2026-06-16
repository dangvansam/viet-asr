# So sánh Kiến trúc Model: Multitask Multitalker ASR Tiếng Việt

**Ngày:** 2026-06-13  
**Phạm vi:** Chỉ bàn về kiến trúc model — không bao gồm data pipeline, preprocessing, hay deployment.  
**Mục tiêu hệ thống:** ASR tiếng Việt với streaming real-time, multitask output (emotion, age, region/dialect, gender, language ID, voice_state, textnorm), EOU detection, CTC forced alignment timestamps, và multitalker (multi-speaker) support.

---

## Mục lục

1. [Tổng quan 3 Kiến trúc Ứng viên](#1-tổng-quan-3-kiến-trúc-ứng-viên)
   - [Fun-ASR-MLT-Nano-2512 (Hybrid Audio-LLM)](#a-fun-asr-mlt-nano-2512-hybrid-audio-llm)
   - [NeMo Parakeet Streaming (repo hiện tại)](#b-nemo-parakeet-streaming-repo-hiện-tại)
   - [Nemotron 3.5 ASR Streaming](#c-nemotron-35-asr-streaming)
2. [Bảng So sánh Chi tiết](#2-bảng-so-sánh-chi-tiết)
3. [Phân tích 5 Hướng Phát triển](#3-phân-tích-5-hướng-phát-triển)
4. [Khuyến Nghị](#4-khuyến-nghị)
5. [Khuyến Nghị Mở Rộng: Universal STT — Kết hợp Tất cả Điểm mạnh](#5-khuyến-nghị-mở-rộng-universal-stt--kết-hợp-tất-cả-điểm-mạnh)
6. [Version A: LLM-base (offline-optimized)](#6-version-a-llm-base-offline-optimized)
7. [Version B: RNNT-base (streaming-optimized)](#7-version-b-rnnt-base-streaming-optimized)

---

## 1. Tổng quan 3 Kiến trúc Ứng viên

### A. Fun-ASR-MLT-Nano-2512 (Hybrid Audio-LLM)

**Source code:** `tmp/FunASR/funasr/models/fun_asr_nano/`  
**Checkpoint:** [FunAudioLLM/Fun-ASR-MLT-Nano-2512](https://hf.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) (arxiv:2509.12508)

> **Lưu ý về biến thể model:** FunASR phát hành 2 checkpoint cùng chia sẻ **chung codebase** `fun_asr_nano` (cùng kiến trúc, cùng paper arxiv:2509.12508):
> | | **Fun-ASR-MLT-Nano-2512** (model mục tiêu) | **Fun-ASR-Nano-2512** (base) |
> |---|---|---|
> | Ngôn ngữ | **31 ngôn ngữ** — thêm nhóm châu Âu (pl, pt, ro, sv, nl, cs, fi, hu...) | 12 ngôn ngữ (zh, en, ja, ko, **yue**, vi, id, th, ms, tl, ar, hi) |
> | Trọng tâm | Multilingual breadth (MLT = Multi-Lingual), translation-oriented | Chinese dialects + rich features (diarization/timestamps/hotwords) |
> | Tokenizer | `multilingual.tiktoken` (vocab rộng hơn) | tiktoken 12-lang |
> | Vietnamese | ✅ Có (cross-lingual transfer rộng) | ✅ Có |
>
> **Kiến trúc dưới đây áp dụng cho cả hai** — khác biệt chỉ ở tokenizer vocab, training data languages, và feature emphasis. MLT có pretraining đa ngôn ngữ rộng hơn → cross-lingual transfer mạnh hơn cho tiếng Việt low-resource, nhưng model card MLT không nhấn mạnh diarization/timestamps/hotwords (cần verify các capability này còn hoạt động trong checkpoint MLT vì code vẫn hỗ trợ).

```
Audio (16kHz)
    │
    ▼
WavFrontend (FBANK 80-dim + CMVN)
    │
    ▼
SenseVoiceEncoderSmall  ◄── frozen (default)
  [SCAMA attention + SinusoidalPositionEncoder]
  [encoders0 (1L) → encoders (N-1 L) → tp_encoders (tp_blocks L)]
    │  encoder_dim (256–768)
    ▼
AudioAdaptor  (Linear | QFormer | Transformer)
  encoder_dim → llm_dim (768)
  optional 2× downsampling (low_frame_rate mode)
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
Qwen3-0.6B LLM (frozen)          CTC Head (Linear)
  [decoder-only causal LM]         encoder_dim → vocab 60515
  [ChatML: system|user|asst]       blank_id = vocab_size - 1
  [vLLM inference engine]          weight = 0.3
    │                                  │
    ▼                                  ▼
Transcription text              CTC Forced Alignment
  (+ multitask via prompt)       (torchaudio.functional.forced_align)
                                  character timestamps, 60ms resolution
```

#### Cơ chế Multitask — SenseVoice Task Token Prepend

Multitask trong FunASR-Nano thực chất đến từ **SenseVoiceEncoder**, không phải từ LLM decoder. SenseVoice prepend 4 task token embeddings vào đầu encoder input sequence:

```python
# sense_voice/model.py:839-865
language_query = self.embed(lids)        # 1 token — language ID (IDs 24884–24992)
style_query    = self.embed(styles)      # 1 token — textnorm flag (25016, 25017)
event_emo_query = self.embed([[1, 2]])   # 2 tokens — emotion + event

# Concatenation order (4 total prepended):
xs = torch.cat([language_query, event_emo_query, xs_pad], dim=1)
# xs_pad = audio frames, language/emo được prepend trước
```

Token ID mapping cụ thể:
| Task | Token IDs | Classes |
|------|-----------|---------|
| Language | 24884(zh), 24885(en), 24888(yue), 24892(ja), 24896(ko), 24992(nospeech) | 6 |
| Emotion | 25001(happy), 25002(sad), 25003(angry), 25004(neutral), 25009(unk) | 5 |
| Event | position 1–2 (generic event tokens) | — |
| Textnorm | 25016(withitn), 25017(woitn) | 2 |

Loss: CTC trên audio frames (bỏ qua 4 token đầu) + CE trên 4 task token positions.

#### Cơ chế Hotword/Context Biasing

```python
# fun_asr_nano/model.py:636-656 — get_prompt()
prompt = "请结合上下文信息，更加准确地完成语音转写任务。\n\n**上下文信息：**\n"
prompt += f"热词列表：[{', '.join(hotwords)}]\n"
prompt += f"语音转写成{language}："

# ChatML format (inference_vllm.py:450-490):
# <|im_start|>system\nYou are a helpful assistant.<|im_end|>
# <|im_start|>user\n{prompt}<|startofspeech|>![audio]<|endofspeech|><|im_end|>
# <|im_start|>assistant\n
```

Hotword hoạt động ở **LLM level** qua prompt injection — không cần model retraining, list hotword thay đổi động được.

#### Cơ chế CTC Forced Alignment

```python
# fun_asr_nano/tools/utils.py:39-73
from torchaudio.functional import forced_align  # Viterbi DP

alignments, scores = forced_align(log_probs, targets, blank=blank_id)
# Frame → token mapping, mỗi token có start_time, end_time, score

# Time conversion (model.py:846-850):
start_ms = frame_idx * 6 * 10 / 1000   # 6ms frame_shift × 10 = 60ms/encoder_frame
```

Resolution: **60ms per encoder frame** → character-level timestamps.

#### Cơ chế Streaming

```python
# inference_vllm_streaming.py: chunk_ms=720, rollback_chars=8
# Strategy: cumulative re-encoding
# - Audio chia thành 720ms chunks
# - MỌI chunks batch vào 1 vLLM generate() call
# - 8 ký tự cuối = "unfixed" (có thể thay đổi ở chunk sau)
# - Output ổn định sau ~3s
```

**Điểm yếu streaming:** Latency ~720ms+, không phải chunk-by-chunk native như RNNT.

#### EOU Detection

Không có explicit EOU detector. EOU = LLM tự generate `<|im_end|>` token khi kết thúc output. Trong streaming, "stable output" = text trước rollback_chars.

#### Training Loss

```
total_loss = llm_cross_entropy + 0.3 × ctc_loss
```

LLM: causal language modeling với teacher forcing. CTC: auxiliary trên encoder frames (bỏ 4 task tokens đầu).

---

### B. NeMo Parakeet Streaming (repo hiện tại)

**Source:** `src/multitalker_asr/`

```
Audio (16kHz)
    │
    ▼
NeMo AudioPreprocessor
  [FilterbankFeatures, 80-dim Mel, 10ms frame shift]
    │
    ▼
Conditioning Layer (pluggable, 4 strategies)
    │
    ▼
StreamingConformerEncoder
  [CacheAwareAttention: context [56 left, 13 right frames]]
  [Chunk profiles: 80ms → 1120ms]
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
RNNT Decoder (BPE 2048)           SortformerEncLabelModel
  [streaming RNNT joint network]     [Speaker diarization: 4 speakers]
    │                                  │
    ▼                                  ▼
Per-speaker transcription         Speaker timestamps + labels
    └──────────────┬───────────────────┘
                   ▼
         Multitalker output
         (speaker-tagged transcription)
```

#### 4 Conditioning Strategies (pluggable)

```python
# configs/conditioning.py:8-12
class ConditioningStrategy(Enum):
    PREPEND_PROMPT_CE  = "prepend_prompt_ce"   # SenseVoice-style
    FEATURE_CONCAT     = "feature_concat"       # Nemotron-style
    DECODER_TAG_ONLY   = "decoder_tag_only"     # tag trong vocab
    HYBRID             = "hybrid"               # concat + tag + aux head
```

**Strategy 1: PREPEND_PROMPT_CE** (SenseVoice-style)
```
[task_pos_0 | task_pos_1 | ... | task_pos_5 | audio_frames...]
  80-dim      80-dim           80-dim
  → Aux CE loss trên 6 positions
  → 6 positions × 80-dim = PromptEmbedding table
```

**Strategy 2: FEATURE_CONCAT** (Nemotron-style)
```
attribute_vector = cat([lang_emb(16), emo_emb(16), gender_emb(8), age_emb(12), region_emb(12)])
                 → broadcast over time → concat along feature axis
                 → Linear projection: (80 + attr_dim) → 80
```

**Strategy 3: DECODER_TAG_ONLY**
```
Target: "<language:vi> <emotion:happy> <gender:female> transcription text"
→ Tags in RNNT BPE vocabulary → no encoder modification
```

**Strategy 4: HYBRID**
```
Feature Concat (encoder input) + Decoder Tags (target) + optional Aux Head
→ Most flexible, highest capacity
```

#### Task Attributes (configs/multitask.py:23-31)

| Task | Classes | Values |
|------|---------|--------|
| language | 4 | vi, en, zh, auto |
| emotion | 7 | happy, sad, angry, neutral, fear, disgust, surprise |
| gender | 2 | male, female |
| age | 4 | child, young, middle_age, old |
| voice_state | 2 | sober, drunk |
| textnorm | 2 | with_itn, without_itn |
| region (optional) | 3 | northern, central, southern |

#### Streaming Architecture

```python
# inference/streaming.py:14-123
# NeMo CacheAwareStreamingAudioBuffer
# Attention context: [56 left, 13 right] frames (default: CHUNK_1120MS)
# Chunk profiles (configs/streaming.py:12-28):
# CHUNK_80MS:   0 right context  → latency ~80ms
# CHUNK_160MS:  1 frame right    → latency ~160ms
# CHUNK_320MS:  3 frames right   → latency ~320ms
# CHUNK_560MS:  6 frames right   → latency ~560ms
# CHUNK_1120MS: 13 frames right  → latency ~1120ms (tốt nhất WER)
```

#### Training Loss

```python
# multitask_model.py:66-69
total_loss = rnnt_loss * 1.0 + ce_aux_loss * ce_loss_weight
# AuxLossScheduler: ce_loss_weight decay về 0 qua 5 epochs
```

#### Điểm Thiếu So với FunASR-Nano

- Không có CTC head → không có character-level timestamps
- Không có EOU detector
- Không có hotword/context biasing mechanism
- NeMo RNNT chỉ có word-level timing (từ blank detection)

---

### C. Nemotron 3.5 ASR Streaming

**Source:** `src/multitalker_asr/data/pipeline/asr_backends/nemotron.py`

```
Audio (16kHz)
    │
    ▼
NeMo AudioPreprocessor (80-dim Mel)
    │
    ▼
StreamingConformerEncoder
  [Nemotron-style per-axis attribute embeddings]
  [Feature-axis concatenation conditioning]
    │
    ▼
RNNT Decoder (BPE)
  [Language tag trong output: <lang:vi>]
    │
    ▼
Multilingual transcription (19 languages, vi-VN included)
```

**Đặc điểm kiến trúc:**
- 0.6B parameters, streaming native
- 19 ngôn ngữ, bao gồm vi-VN
- Language conditioning qua Decoder Tag strategy
- Per-axis attribute embeddings (Nemotron-style) cho language routing
- **Không có:** emotion, age, gender, region heads
- Proprietary checkpoint từ NVIDIA

---

## 2. Bảng So sánh Chi tiết

| Tiêu chí | Fun-ASR-MLT-Nano-2512 | NeMo Parakeet (hiện tại) | Nemotron 3.5 |
|----------|:---------------:|:------------------------:|:------------:|
| **Encoder** | SenseVoiceEncoder (SCAMA) | StreamingConformer (NeMo) | StreamingConformer (NeMo) |
| **Decoder** | Qwen3-0.6B LLM (causal) | RNNT (BPE 2048) | RNNT (BPE) |
| **Decoder type** | Auto-regressive LLM | Transducer (streaming) | Transducer (streaming) |
| **Streaming native** | Không (batch chunks) | Có | Có |
| **Streaming latency** | ~720ms+ (min) | 80–1120ms (configurable) | 80–1120ms |
| **EOU Detection** | Implicit (LLM `<im_end>`) | Không | Không |
| **Multitask: Language** | Có (31 languages, MLT vocab) | Có (4 classes) | Có (19 languages) |
| **Multitask: Emotion** | Có (4 classes) | Có (7 classes) | Không |
| **Multitask: Gender** | Không | Có (2 classes) | Không |
| **Multitask: Age** | Không | Có (4 classes) | Không |
| **Multitask: Region** | Không | Có (3 classes) | Không |
| **Multitask: Voice State** | Không | Có (2 classes) | Không |
| **Hotword / Context Biasing** | Có (LLM prompt injection) | Không | Không |
| **CTC Forced Alignment** | Có (character-level, 60ms) | Không | Không |
| **Multitalker (multi-speaker)** | Không | Có (4 speakers, Sortformer) | Không |
| **Vietnamese support** | Tốt (SenseVoice + 31-lang transfer) | Chưa rõ (Parakeet multilingual) | Có (vi-VN) |
| **GPU Memory** | Cao (vLLM + LLM 0.6B) | Vừa | Vừa |
| **Inference throughput** | Cao (vLLM KV cache) | Trung bình | Trung bình |
| **Finetune difficulty** | Trung bình (encoder frozen) | Dễ (NeMo tooling) | Dễ (NeMo tooling) |
| **Train từ đầu** | Khó (LLM dependency) | Khả thi | Khả thi |
| **Open weights** | Có (FunASR Hub) | Có (NVIDIA NGC) | Có (NVIDIA NGC) |
| **Framework** | FunASR + HuggingFace | NeMo + PyTorch Lightning | NeMo + PyTorch Lightning |

---

## 3. Phân tích 5 Hướng Phát triển

### Option 1: Finetune Fun-ASR-MLT-Nano-2512 (full hoặc encoder-only)

```
Giữ nguyên kiến trúc Fun-ASR-MLT-Nano, finetune trên dữ liệu tiếng Việt
```

**Ưu điểm:**
- LLM decoder (Qwen3) mạnh, xử lý tốt ambiguity và context
- Hotword miễn phí qua prompt injection — không cần retraining khi thêm hotword
- CTC timestamps character-level chính xác (60ms resolution)
- SenseVoice encoder đã proven tốt cho Vietnamese (thực nghiệm nội bộ)
- **31-language pretraining → cross-lingual transfer mạnh cho tiếng Việt low-resource**
- vLLM cho throughput cao khi batch inference

**Nhược điểm:**
- **Streaming latency ~720ms+** — không thể xuống dưới 720ms với thiết kế hiện tại
- **Không native multitalker** — FunASR-Nano là single-talker, thêm multitalker rất phức tạp
- Phụ thuộc vLLM → deployment phức tạp hơn, GPU memory cao hơn
- LLM frozen → chỉ có thể adapter-train hoặc train audio side
- Age, region, gender **không có sẵn** trong SenseVoice task tokens
- EOU không explicit, phụ thuộc vào LLM generation behavior

**Phù hợp khi:** Ưu tiên chất lượng transcription + timestamps, chấp nhận latency > 1s, single-talker.

---

### Option 2: SenseVoiceEncoder + NeMo RNNT Decoder (Hybrid)

```
Audio → SenseVoiceEncoder (FunASR, finetune) → NeMo RNNT Decoder
                    │                                │
              CTC Head (thêm mới)            Sortformer Diarization
              [timestamps]                   [multitalker]
```

**Ưu điểm:**
- Tận dụng SenseVoice encoder mạnh nhất cho tiếng Việt
- RNNT native streaming: latency 80–160ms
- Giữ được Sortformer multitalker support từ NeMo
- CTC head thêm được (linear projection từ encoder) → character timestamps
- Multitask conditioning đầy đủ 6-7 attributes (repo đã có 4 strategies)
- Có thể extend SenseVoice task tokens với Age, Region, Gender tokens
- EOU qua RNNT blank detection + optional CTC blank-ratio threshold

**Nhược điểm:**
- Hai framework (FunASR + NeMo) → integration complexity
- SenseVoice encoder output format cần adapt sang NeMo RNNT input format
- Không có hotword mechanism ở RNNT level (chỉ shallow fusion n-gram)
- Cần custom training loop để align hai framework

**Phù hợp khi:** Cần low-latency streaming + chất lượng Vietnamese tốt + multitalker.

---

### Option 3: Mở rộng Repo Hiện tại (NeMo Parakeet + FunASR Improvements)

```
Giữ NeMo Parakeet làm backbone, thêm các cải tiến từ FunASR
```

**Cải tiến thêm vào:**
- **CTC auxiliary head** (Linear: encoder_dim → ctc_vocab_size) sau encoder → timestamps
- **Explicit EOU module** (CTC blank-ratio threshold hoặc RNNT silence detection)
- Nâng số emotion classes từ 7 lên, thêm region/dialect labels
- Cải thiện conditioning với SenseVoice-style multi-position prepend

**Ưu điểm:**
- Rủi ro thấp nhất — NeMo tooling quen thuộc, không thay đổi kiến trúc lớn
- Multitalker (Sortformer) đã hoạt động tốt
- Streaming profiles đã được calibrate
- Thêm CTC head chỉ là +1 Linear layer + CTCLoss trong training

**Nhược điểm:**
- Encoder (NeMo Conformer) không mạnh bằng SenseVoice encoder cho Vietnamese
- Không có LLM-level hotword/context biasing
- WER trên tiếng Việt có thể thua FunASR-Nano do encoder yếu hơn
- Phải train từ đầu hoặc finetune Parakeet encoder trên Vietnamese data

**Phù hợp khi:** Dữ liệu hạn chế (< 500h), muốn incremental improvement, timeline ngắn.

---

### Option 4: Finetune Nemotron 3.5 + Thêm Multitask Heads

```
Nemotron 3.5 streaming (frozen encoder) + thêm auxiliary classification heads
```

**Thêm vào:**
- Auxiliary heads sau encoder: Emotion, Age, Gender, Region classifiers
- Dùng PREPEND_PROMPT_CE hoặc FEATURE_CONCAT conditioning từ repo hiện tại
- Finetune trên Vietnamese data

**Ưu điểm:**
- Nemotron 3.5 proven Vietnamese support (vi-VN trong 19 languages)
- Streaming native với low latency
- Checkpoint từ NVIDIA — quality baseline tốt

**Nhược điểm:**
- Thêm task heads **sau** backbone không mạnh bằng SenseVoice's token injection (task-aware encoder representations)
- Proprietary checkpoint — giới hạn về commercial use, community sharing
- Age, Region, Gender heads phải train từ đầu với ít supervised data
- Không có CTC timestamps (Nemotron là RNNT-only)
- Cần giải phóng encoder weights từ NVIDIA để finetune hiệu quả

**Phù hợp khi:** Muốn nhanh, có Vietnamese data nhỏ, không cần timestamps.

---

### Option 5: Kiến trúc Mới Từ Đầu (Best of Both Worlds)

```
Audio → WavFrontend (FunASR-style)
    │
    ▼
[lang | emo | age | region | gender | voice_state] PREPEND (6 task tokens)
    │
    ▼
SenseVoiceEncoderSmall (finetune từ pretrained, SCAMA attention)
    │
    ├─────────────────────────┐
    ▼                         ▼
CTC Head                 StreamingRNNT Decoder
(Linear proj)            (BPE vocab, NeMo-style)
[char timestamps]        [transcription + decoder tags]
    │                         │
    └──────────┬───────────────┘
               ▼
    + Sortformer Diarization (NeMo)
               ▼
    Multitalker output với timestamps + all task attributes
```

**Đặc điểm kiến trúc:**
- Task token injection: 6 positions prepend trước audio (SenseVoice-style)
- Extend SenseVoice vocab với Age/Region/Gender/VoiceState tokens
- Dual-head: CTC (timestamps) + RNNT (transcription)
- Conditioning: HYBRID strategy từ repo hiện tại
- EOU: CTC blank-ratio + RNNT silence + optional VAD signal fusion

**Loss:**
```
total_loss = rnnt_loss * 1.0 + ctc_loss * 0.3 + task_ce_loss * λ(t)
# λ(t) = aux_loss_scheduler.step() → decay qua 5 epochs
```

**Ưu điểm:**
- Full control over architecture
- Tối ưu cho tiếng Việt từ đầu
- CTC timestamps + RNNT streaming + multitalker + multitask đầy đủ
- Không phụ thuộc vào proprietary checkpoints sau khi train

**Nhược điểm:**
- Cần dữ liệu lớn với tất cả task labels (emotion, age, gender, region với timestamp)
- Training cost cao (encoder + RNNT + CTC head + task tokens)
- Rủi ro cao nhất — nhiều moving parts, debugging phức tạp
- Timeline dài nhất

**Phù hợp khi:** > 1000h Vietnamese labeled data, GPU cluster, muốn production-grade system không phụ thuộc bên ngoài.

---

## 4. Khuyến Nghị

### Lựa Chọn Theo Điều Kiện

| Điều kiện | Khuyến nghị | Lý do chính |
|-----------|-------------|-------------|
| Data < 500h, timeline ngắn | **Option 3** | Rủi ro thấp, NeMo tooling sẵn có |
| Data 500h–1000h, cần chất lượng | **Option 2** | SenseVoice encoder mạnh + RNNT streaming |
| Data > 1000h + GPU cluster | **Option 5** | Full control, tối ưu nhất |
| Chỉ cần timestamps (no streaming) | **Option 1** | CTC align tốt nhất + hotword free |
| Cần multilingual nhanh | **Option 4** | Nemotron proven, ít custom |

---

### Khuyến Nghị Chính: **Option 2 — SenseVoiceEncoder + NeMo RNNT Decoder**

**Lý do chi tiết:**

**1. Vietnamese encoder quality:**
SenseVoiceEncoder của Fun-ASR-MLT-Nano đã được probe thực tế trên Vietnamese data (Qwen3-ForcedAligner experiment, 2026-06-09 — 13 monotonic word spans với correct text). MLT variant train trên 31 ngôn ngữ với hàng chục triệu giờ audio → cross-lingual transfer rộng, lợi cho tiếng Việt low-resource. NeMo Conformer trong Parakeet chưa được confirm performance trên tiếng Việt.

**2. Streaming latency:**
RNNT native streaming: 80–160ms với chunk profile CHUNK_160MS.  
FunASR-Nano streaming: minimum 720ms do cumulative re-encoding strategy.  
Đây là gap 4-9× latency — quyết định với real-time application.

**3. Multitalker support:**
Sortformer diarization (NeMo) là critical feature. FunASR-Nano hoàn toàn không có multitalker support. Option 2 giữ được Sortformer khi kết nối qua NeMo decoder side.

**4. CTC timestamps:**
Thêm CTC head (1 Linear layer: `encoder_dim → ctc_vocab_size`) sau SenseVoiceEncoder là minimal cost — dùng `torchaudio.functional.forced_align()` giống FunASR pattern. Character-level timestamps với 60ms resolution.

**5. Multitask richness:**
Repo hiện tại có 6–7 task attributes (emotion 7 classes, region 3 classes) vs SenseVoice chỉ có 4 (emotion 4 classes, no age/gender/region). Option 2 có thể extend SenseVoice vocab với new task tokens và finetune trên Vietnamese labeled data.

**6. EOU detection:**
Combine CTC blank-ratio threshold (khi >X% frames predict blank → silence → EOU) với RNNT endpoint detection. Robust hơn LLM implicit EOU.

**7. Hotword (acceptable tradeoff):**
Mất LLM-level prompt injection nhưng có thể implement RNNT n-gram shallow fusion hoặc trie-based hotword boosting — đủ cho production use case.

---

### Implementation Roadmap cho Option 2

```
Phase 1: SenseVoiceEncoder → NeMo Interface Adapter
  - Wrap SenseVoiceEncoderSmall với NeMo AbstractEncoder interface
  - Map encoder output format: (B, T, encoder_dim) → NeMo encoder output format
  - Verify CacheAware streaming compatibility với SCAMA attention

Phase 2: CTC Head
  - Add Linear(encoder_dim, ctc_vocab_size) + CTCLoss
  - Implement forced_align() wrapper (torchaudio.functional)
  - Integrate vào MultitaskModel.forward()

Phase 3: Task Token Extension
  - Extend SenseVoice vocab với Age (4), Gender (2), Region (3), VoiceState (2) tokens
  - Finetune token embeddings trên Vietnamese labeled data
  - Add new task positions vào PromptEmbedding table

Phase 4: EOU Module
  - CTC blank-ratio detector: sliding window blank fraction > threshold
  - Fuse với RNNT silence detection
  - Optional: lightweight VAD signal as third input

Phase 5: End-to-end Finetune
  - Finetune SenseVoice encoder (partial unfreeze: top 4 layers) + train CTC/RNNT heads
  - AuxLossScheduler cho task token CE loss
  - Evaluate: WER, task accuracy, timestamp error (ms), EOU precision/recall
```

---

### Ghi chú về Kiến trúc Tương lai

Nếu sau này muốn nâng cấp lên **LLM decoder** (vì quality tốt hơn RNNT ở offline mode), architecture Option 2 cho phép **dual-mode inference:**
- Streaming: SenseVoice encoder → RNNT decoder (low latency)  
- Offline/high-accuracy: SenseVoice encoder → AudioAdaptor → Qwen3-0.6B (FunASR-Nano style)

Cùng một encoder, hai inference paths — encoder weights shared.

---

## 5. Khuyến Nghị Mở Rộng: Universal STT — Kết hợp Tất cả Điểm mạnh

> Đây là tầm nhìn dài hạn cho một model STT **đa năng (one model, all capabilities)**. Về bản chất là Option 5 nâng cấp tối đa: một encoder dùng chung, nhiều decode path + nhiều head, gộp mọi điểm mạnh của 3 kiến trúc đã phân tích.

### Triết lý thiết kế

| Điểm mạnh | Lấy từ | Cách tích hợp |
|-----------|--------|---------------|
| Encoder đa ngôn ngữ mạnh | Fun-ASR-MLT (SenseVoice, 31-lang) | Shared backbone, partial finetune |
| Streaming latency thấp | NeMo RNNT (cache-aware) | RNNT decode path |
| Offline accuracy cao | Qwen3-0.6B LLM | LLM decode path (dual-mode) |
| CTC timestamps | Fun-ASR-Nano CTC head | CTC head song song |
| Multitalker | NeMo Sortformer | Diarization branch |
| Multitask đầy đủ (emotion/age/gender/region/voice_state) | repo hiện tại (4 strategies) | Task token prepend + feature concat |
| Hotword/context biasing | Fun-ASR prompt injection + RNNT shallow fusion | LLM prompt (offline) + n-gram trie (streaming) |
| EOU detection | CTC blank-ratio + RNNT endpoint | Fusion module |

### Kiến trúc Tổng thể

```
                          Audio (16kHz)
                               │
                               ▼
                        WavFrontend (FBANK 80-dim + CMVN)
                               │
       ┌───────────────────────┼────────────────────────┐
       ▼                       ▼                         ▼
[task token prepend]    SHARED ENCODER             [hotword trie /
 lang|emo|age|         SenseVoiceEncoderSmall       n-gram FST]
 gender|region|        (31-lang, finetuned)              │
 voice_state          + feature-concat conditioning      │ (shallow fusion
       │               (Nemotron-style attr embed)        │  ở streaming)
       └───────────────────────┤                          │
                               │ encoder_out (B,T,D)       │
        ┌──────────────┬───────┼───────────┬──────────────┤
        ▼              ▼       ▼           ▼              ▼
   ┌─────────┐   ┌──────────┐ ┌────────┐ ┌──────────┐ ┌──────────┐
   │ CTC Head│   │  RNNT    │ │ LLM    │ │ Task     │ │Sortformer│
   │(Linear) │   │ Decoder  │ │ Decoder│ │ Heads    │ │ Diarize  │
   │timestamps│  │(streaming│ │(offline│ │(emo/age/ │ │(4 spk)   │
   │ 60ms res │  │ 80-160ms)│ │ Qwen3) │ │ gender..)│ │          │
   └────┬─────┘  └────┬─────┘ └───┬────┘ └────┬─────┘ └────┬─────┘
        │             │           │           │            │
        ▼             ▼           ▼           ▼            ▼
   timestamps    streaming    offline     attribute   speaker-tagged
                  text       hi-acc text    tags        segments
        │             │           │           │            │
        └─────────────┴───────────┴───────────┴────────────┘
                               │
                               ▼
                     EOU Fusion Module
              (CTC blank-ratio + RNNT endpoint + VAD)
                               │
                               ▼
        Unified Output: {speaker, text, timestamps,
                         emotion, age, gender, region,
                         voice_state, language, is_eou}
```

### Cơ chế Inference Đa chế độ (mode routing)

Một model, người dùng chọn path theo use-case:

| Mode | Path active | Latency | Use-case |
|------|-------------|---------|----------|
| **Real-time streaming** | Encoder → RNNT + CTC + Task heads | 80–160ms | Live caption, voice assistant |
| **Offline high-accuracy** | Encoder → LLM + CTC + Task heads | ~giây | Transcription chất lượng cao, subtitle |
| **Multitalker meeting** | Encoder → Sortformer → RNNT per-speaker | ~chunk | Họp, phỏng vấn nhiều người |
| **Rich analysis** | Tất cả heads | varies | Phân tích cảm xúc/nhân khẩu học |

Encoder weights **shared 100%** giữa các path → chỉ load 1 backbone, swap decode head theo nhu cầu.

### Training Strategy (multi-objective)

```
total_loss = w_rnnt · L_rnnt          (streaming transcription)
           + w_ctc  · L_ctc           (timestamps, 0.3)
           + w_llm  · L_llm_ce        (offline transcription, optional joint)
           + w_task · L_task_ce · λ(t) (multitask, aux scheduler decay)
           + w_diar · L_sortformer    (diarization, có thể train riêng)
```

**Curriculum đề xuất (giảm rủi ro multi-objective):**
1. **Stage 1** — Finetune encoder + RNNT trên Vietnamese ASR (single objective, ổn định backbone)
2. **Stage 2** — Freeze encoder, add CTC head + task token heads (timestamps + attributes)
3. **Stage 3** — Add LLM decode path (chia sẻ encoder, train adaptor + LoRA trên Qwen3)
4. **Stage 4** — Integrate Sortformer (train riêng hoặc joint với frozen ASR)
5. **Stage 5** — Joint finetune nhẹ tất cả heads với loss weights cân bằng

### Trade-offs cần biết

**Ưu điểm:**
- Một model phục vụ mọi nhu cầu — giảm chi phí maintain, deploy, version
- Encoder shared → tiết kiệm GPU memory vs chạy nhiều model riêng
- Cross-task transfer: học emotion có thể cải thiện ASR và ngược lại
- Future-proof: thêm head mới (ví dụ: language translation) không phá vỡ kiến trúc

**Nhược điểm / rủi ro:**
- **Phức tạp nhất** — nhiều moving parts, debugging khó, training pipeline dài
- Cần dữ liệu đa nhãn lớn (ASR + emotion + age + gender + region + speaker + timestamp) — hiếm khi có đủ trên 1 dataset, phải mix nhiều nguồn
- Risk **negative transfer**: task phụ kéo WER chính xuống nếu loss weight sai
- LLM + RNNT + Sortformer cùng lúc → GPU memory cao khi train (nhưng có thể train staged)
- Maintenance: thay đổi 1 head có thể ảnh hưởng shared encoder

### Lộ trình Thực dụng

Không cần build tất cả cùng lúc. Khởi đầu từ **Option 2 (SenseVoiceEncoder + RNNT)** rồi mở rộng dần thành Universal STT qua các stage curriculum ở trên. Mỗi stage cho một sản phẩm dùng được — không phải all-or-nothing:

```
Option 2 (MVP: streaming + multitalker + multitask cơ bản)
   └─► + CTC head        → timestamps
        └─► + task heads  → emotion/age/gender/region đầy đủ
             └─► + LLM path → offline high-accuracy mode
                  └─► + hotword fusion → context-aware
                       └─► Universal STT (đầy đủ)
```

**Kết luận:** Universal STT là **đích đến**, Option 2 là **điểm khởi đầu**. Kiến trúc shared-encoder + multi-head cho phép tiến hóa từng bước mà không phải viết lại từ đầu — đây chính là lý do Option 2 được khuyến nghị làm nền móng.

---

## Chiến lược 3-Model Family (Section 6 + 7)

Universal STT (Section 5) là flagship đầy đủ heads — tốt cho R&D và rich-analysis, nhưng **nặng để deploy production** vì gánh cả LLM + RNNT + Sortformer cùng lúc. Giải pháp thực dụng: từ **encoder backbone dùng chung**, chưng cất (distill/specialize) ra **2 phiên bản production gọn**, mỗi version bias nhẹ về một nhu cầu:

```
                  ┌─────────────────────────────────┐
                  │   SHARED ENCODER (Section 5)     │
                  │   SenseVoiceEncoder 31-lang      │
                  │   + task conditioning            │
                  │   ── train 1 lần, weights shared ─┤
                  └────────────┬────────────────────┘
                               │ encoder weights khởi tạo cho cả 2
              ┌────────────────┴─────────────────┐
              ▼                                  ▼
   ┌──────────────────────┐         ┌──────────────────────┐
   │ Version A (Sec 6)     │         │ Version B (Sec 7)     │
   │ LLM-base / FunASR-like│         │ RNNT-base / NeMo-like │
   │ → OFFLINE accuracy    │         │ → STREAMING latency   │
   └──────────────────────┘         └──────────────────────┘

Tổng: 3 model — 1 Universal (flagship) + 2 production-specialized.
Cả 3 chia sẻ encoder backbone → train encoder 1 lần, tái sử dụng.
```

**Nguyên tắc "kế thừa mạnh + biasing nhỏ":** Cả 2 version giữ TẤT CẢ điểm mạnh (encoder 31-lang, multitask, timestamps, multitalker, hotword). Chỉ khác ở **decode path chính** và **vài siêu tham số bias** theo use-case — không phải hai kiến trúc tách rời.

---

## 6. Version A: LLM-base (offline-optimized)

> Kế thừa Fun-ASR-MLT-Nano. Tối ưu cho **độ chính xác offline** — subtitle, transcription service, meeting notes, dữ liệu chất lượng cao.

### Kiến trúc

```
Audio → WavFrontend → SHARED SenseVoiceEncoder (full-context, KHÔNG streaming mask)
                            │  [task token prepend: lang|emo|age|gender|region]
                            ├──────────────────────────┐
                            ▼                          ▼
                    AudioAdaptor                  CTC Head
                  (Linear/QFormer)              (timestamps 60ms)
                            │
                            ▼
                    Qwen3-0.6B LLM (LoRA finetune)
                  [ChatML + hotword prompt injection]
                  [vLLM batch inference, beam/sampling]
                            │
                            ▼
              text (hi-accuracy) + timestamps + tags
                            +
              Sortformer diarization (offline, full-audio) → multitalker
```

### Biasing chuyên biệt cho offline

| Khía cạnh | Cấu hình bias | Lý do |
|-----------|---------------|-------|
| Encoder attention | **Full bidirectional context** (bỏ streaming mask) | Nhìn toàn câu → accuracy cao nhất |
| Decode | LLM với **beam search / sampling**, max_tokens lớn | Quality > latency |
| Hotword | **Prompt injection** (LLM-level) — mạnh nhất | Context-aware đầy đủ |
| Diarization | Sortformer chạy trên **toàn audio** (offline) | Speaker assignment chính xác |
| Batch | vLLM **batch nhiều utterance** | Throughput cao |
| CTC | Forced alignment **sau khi có full transcript** | Timestamps chính xác character-level |

### Điểm mạnh kế thừa
- Encoder 31-lang (cross-lingual transfer cho tiếng Việt)
- LLM decoder → xử lý ambiguity, ngữ cảnh dài, ITN tốt nhất
- Hotword prompt injection miễn phí
- CTC timestamps + Sortformer multitalker + full multitask heads

### Trade-off chấp nhận
- Latency cao (giây) — **không dùng cho real-time**, nhưng offline không cần
- GPU memory cao (vLLM + LLM) — bù lại bằng batch throughput

### Use-case
Subtitle phim/video, transcription dịch vụ, biên bản họp, data labeling, bất kỳ tác vụ ưu tiên WER thấp nhất.

---

## 7. Version B: RNNT-base (streaming-optimized)

> Kế thừa NeMo Parakeet. Tối ưu cho **độ trễ thấp real-time** — live caption, voice assistant, phụ đề trực tiếp.

### Kiến trúc

```
Audio (chunk) → WavFrontend → SHARED SenseVoiceEncoder (cache-aware streaming mask)
                                   │  [feature-concat conditioning: attr embed broadcast]
                                   ├──────────────┬──────────────┐
                                   ▼              ▼              ▼
                            RNNT Decoder    CTC Head      Lightweight Task Heads
                          (streaming joint) (blank-ratio   (emo/age/gender/region,
                          [cache-aware]      → EOU)         frame-pooled, nhẹ)
                                   │              │              │
                                   ▼              ▼              ▼
                          streaming text    EOU signal     attribute tags
                                   │
                                   ▼
                    Sortformer streaming diarization → multitalker real-time
```

### Biasing chuyên biệt cho streaming

| Khía cạnh | Cấu hình bias | Lý do |
|-----------|---------------|-------|
| Encoder attention | **Cache-aware streaming mask** [56 left, 13 right] | Latency 80–160ms |
| Decode | RNNT **greedy** (không beam) | Tốc độ real-time |
| Chunk | Profile nhỏ (CHUNK_160MS / 320MS) | Trade WER lấy latency |
| EOU | **CTC blank-ratio + RNNT endpoint** — first-class | Voice assistant cần turn-taking |
| Task heads | **Lightweight** (frame-pooled, ít param) | Không cản throughput |
| Hotword | **RNNT shallow fusion / n-gram trie** (không LLM) | Bias streaming không cần LLM |
| Diarization | Sortformer **streaming** mode | Multitalker real-time |

### Điểm mạnh kế thừa
- Encoder 31-lang dùng chung (cùng backbone với Version A)
- RNNT native streaming → latency thấp nhất
- EOU detection first-class (quan trọng cho voice agent)
- Multitalker real-time (Sortformer streaming) + multitask nhẹ + CTC timestamps

### Trade-off chấp nhận
- WER hơi cao hơn Version A (greedy RNNT vs LLM beam) — chấp nhận được cho real-time
- Hotword yếu hơn LLM prompt (shallow fusion) — đủ cho từ khóa domain

### Use-case
Live caption hội nghị, voice assistant tiếng Việt, phụ đề livestream, call-center real-time, bất kỳ tác vụ cần phản hồi tức thì.

---

## Tổng kết 3-Model Family

| | Universal (Sec 5) | Version A — LLM (Sec 6) | Version B — RNNT (Sec 7) |
|---|:---:|:---:|:---:|
| **Vai trò** | Flagship / R&D | Production offline | Production streaming |
| **Decode chính** | RNNT + LLM + tất cả heads | LLM (Qwen3) | RNNT |
| **Encoder** | Shared backbone | Shared (full-context) | Shared (streaming mask) |
| **Tối ưu cho** | Mọi capability | WER thấp nhất | Latency thấp nhất |
| **Latency** | varies | giây (offline) | 80–160ms |
| **Hotword** | Cả hai cơ chế | Prompt injection | Shallow fusion |
| **EOU** | Có | Có (offline ít cần) | First-class |
| **Deploy weight** | Nặng | Vừa (LLM) | Nhẹ (RNNT) |

**Workflow phát triển:**
1. Train **shared encoder** một lần (Stage 1–2 curriculum ở Section 5) trên Vietnamese 31-lang data
2. Fork ra **Version A**: gắn LLM decode path + LoRA finetune, bias full-context
3. Fork ra **Version B**: gắn RNNT decode path + streaming mask, bias cache-aware
4. (Optional) Universal = giữ cả hai path cho R&D / rich-analysis

Encoder weights tái sử dụng → **train tốn kém 1 lần, sinh ra cả 3 model**. Đây là lợi thế lớn nhất của thiết kế shared-backbone.

---

*References:*
- *Checkpoint: [FunAudioLLM/Fun-ASR-MLT-Nano-2512](https://hf.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) (arxiv:2509.12508)*
- *`tmp/FunASR/funasr/models/fun_asr_nano/inference_vllm.py`*
- *`tmp/FunASR/funasr/models/sense_voice/model.py`*
- *`src/multitalker_asr/models/multitask_model.py`*
- *`src/multitalker_asr/configs/multitask.py`*
- *`src/multitalker_asr/inference/streaming.py`*
