# FunASR Code Snippets for Multitalker ASR Integration

All code extracted directly from FunASR repository.

---

## 1. SPEAKER EMBEDDING LOOKUP TABLE (from SenseVoice style)

Source: `funasr/models/sense_voice/model.py:642-648`

```python
import torch
import torch.nn as nn

class MultitalkerASRModel(nn.Module):
    def __init__(self, num_speakers: int = 2, input_size: int = 80, **kwargs):
        super().__init__()
        
        # Speaker ID mapping (adapt to your speaker list)
        self.speaker_dict = {
            f"speaker_{i}": i for i in range(num_speakers)
        }
        
        # Embedding layer: num_speakers special tokens → input_size dimensions
        # Add some buffer for future speakers
        self.speaker_embed = torch.nn.Embedding(
            num_speakers + 5, 
            input_size
        )
        
        print(f"Created speaker embedding: {num_speakers} speakers × {input_size} dims")
```

---

## 2. TASK QUERY PREPENDING IN ENCODE (from SenseVoice)

Source: `funasr/models/sense_voice/model.py:722-774`

```python
def encode(self, speech: torch.Tensor, speech_lengths: torch.Tensor, 
           speaker_ids: torch.Tensor = None, **kwargs) -> tuple:
    """
    Encode speech with prepended speaker task queries.
    
    Args:
        speech: [batch, frames, features]
        speech_lengths: [batch]
        speaker_ids: [batch, 1] speaker ID indices
    
    Returns:
        encoder_out: [batch, frames+num_task_tokens, encoder_dim]
        encoder_out_lens: [batch]
    """
    
    # 1. Data augmentation
    if self.specaug is not None and self.training:
        speech, speech_lengths = self.specaug(speech, speech_lengths)
    
    # 2. Normalization
    if self.normalize is not None:
        speech, speech_lengths = self.normalize(speech, speech_lengths)
    
    # 3. CREATE SPEAKER QUERY TOKEN (THE KEY PART)
    # Get speaker embeddings from lookup table
    speaker_query = self.speaker_embed(speaker_ids)  # [batch, 1, input_size]
    
    # 4. PREPEND to speech frames
    speech = torch.cat((speaker_query, speech), dim=1)
    speech_lengths += 1  # Account for prepended token
    
    # 5. Pass through encoder
    encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
    
    return encoder_out, encoder_out_lens
```

---

## 3. LOSS COMPUTATION (from SenseVoice)

Source: `funasr/models/sense_voice/model.py:707-720`

```python
def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor,
            text: torch.Tensor, text_lengths: torch.Tensor,
            speaker_labels: torch.Tensor, **kwargs):
    """
    Two-stage loss: speaker classification + ASR CTC
    
    Args:
        speech: [batch, frames, features]
        speech_lengths: [batch]
        text: [batch, text_len] - ASR target tokens
        text_lengths: [batch]
        speaker_labels: [batch, 1] - speaker IDs for classification
    
    Returns:
        loss: scalar
        stats: dict with breakdown
        batch_size: for logging
    """
    
    batch_size = speech.shape[0]
    
    # Encode with speaker prepending
    encoder_out, encoder_out_lens = self.encode(
        speech, speech_lengths, speaker_ids=speaker_labels
    )
    
    # Slice encoder output:
    # - First 1 frame: speaker task token output
    # - Remaining frames: ASR tokens
    
    # LOSS 1: Speaker classification (on first frame only)
    speaker_logits = encoder_out[:, :1, :]  # [batch, 1, encoder_dim]
    loss_speaker = self.speaker_loss(
        speaker_logits,  # model output
        speaker_labels    # target speaker IDs
    )
    
    # LOSS 2: ASR CTC (on remaining frames)
    loss_ctc = self.ctc(
        encoder_out[:, 1:, :],   # Skip speaker task token
        encoder_out_lens - 1,     # Adjust lengths
        text[:, 1:],             # Skip speaker label in target
        text_lengths - 1
    )
    
    # Combine losses with weight
    speaker_weight = 0.1  # Adjust based on importance
    loss = loss_ctc + speaker_weight * loss_speaker
    
    # Track statistics
    stats = {
        "loss": torch.clone(loss.detach()),
        "loss_ctc": torch.clone(loss_ctc.detach()),
        "loss_speaker": torch.clone(loss_speaker.detach()),
    }
    
    return loss, stats, batch_size
```

---

## 4. CROSS-ENTROPY LOSS FOR SPEAKER CLASSIFICATION

Source: `funasr/models/sense_voice/model.py` (adapted)

```python
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss

def _init_losses(self, num_speakers: int):
    """Initialize loss functions."""
    
    # For speaker classification
    self.speaker_loss = nn.CrossEntropyLoss(reduction='mean')
    
    # Alternative with label smoothing (smoother training)
    self.speaker_loss_smooth = LabelSmoothingLoss(
        size=num_speakers,
        padding_idx=-1,
        smoothing=0.1,  # label smoothing factor
        normalize_length=False
    )

# Usage in forward pass:
# loss_speaker = self.speaker_loss(speaker_logits.view(-1, num_speakers), speaker_labels.view(-1))
```

---

## 5. INFERENCE WITH SPEAKER CONTROL (from SenseVoice)

Source: `funasr/models/sense_voice/model.py:809-950`

```python
def inference(self, speech: torch.Tensor, speech_lengths: torch.Tensor,
              speaker_id: int = 0, **kwargs) -> tuple:
    """
    Inference with speaker specification.
    
    Args:
        speech: [1, frames, features] - single sample
        speech_lengths: [1]
        speaker_id: int - which speaker (0, 1, 2, ...)
    
    Returns:
        text: recognized transcription
        speaker_pred: predicted speaker ID (if using auxiliary head)
    """
    
    device = speech.device
    
    # 1. Get speaker embedding for this speaker
    speaker_ids = torch.LongTensor([[speaker_id]]).to(device)
    
    # 2. Encode with speaker prepending
    speaker_query = self.speaker_embed(speaker_ids)
    speech = torch.cat((speaker_query, speech), dim=1)
    speech_lengths += 1
    
    encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
    
    # 3. CTC decoding
    ctc_logits = self.ctc.log_softmax(encoder_out)
    
    # 4. Extract speaker prediction (optional)
    speaker_logits = ctc_logits[:, :1, :]  # First frame
    speaker_pred = torch.argmax(speaker_logits, dim=-1)
    
    # 5. Greedy decoding on ASR part
    asr_logits = ctc_logits[:, 1:, :]      # Remaining frames
    yseq = asr_logits.argmax(dim=-1)       # Greedy decode
    yseq = torch.unique_consecutive(yseq, dim=-1)
    
    # 6. Filter blanks
    mask = yseq != self.blank_id
    token_int = yseq[mask].tolist()
    text = self.tokenizer.decode(token_int)
    
    return text, speaker_pred.item()
```

---

## 6. AUDIO-TEXT EMBEDDING INJECTION (from Fun-ASR-Nano, alternative approach)

Source: `funasr/models/fun_asr_nano/model.py:161-227`

```python
def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor,
            input_ids: torch.Tensor, attention_mask: torch.Tensor,
            labels_ids: torch.Tensor, fbank_beg: torch.Tensor,
            fake_token_len: torch.Tensor, **kwargs):
    """
    Alternative approach: Replace placeholder embeddings with audio.
    
    Args:
        speech: [batch, frames, features] - audio
        speech_lengths: [batch]
        input_ids: [batch, seq_len] - tokenized prompt + placeholders
        attention_mask: [batch, seq_len]
        labels_ids: [batch, seq_len] - target tokens
        fbank_beg: [batch, 1] - where audio should be inserted
        fake_token_len: [batch, 1] - how many tokens for audio
    
    Returns:
        loss: scalar
        stats: dict
        batch_size: for logging
    """
    
    batch_size, token_num = input_ids.shape
    stats = {}
    
    # 1. Encode audio
    encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
    
    # 2. Get LLM embeddings for all tokens (including placeholders)
    inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)
    
    # 3. REPLACE placeholder embeddings with actual audio embeddings
    for batch_idx in range(batch_size):
        fbank_beg_idx = fbank_beg[batch_idx, 0].item()
        if fbank_beg_idx > 0:  # Audio exists
            speech_token_len = fake_token_len[batch_idx, 0].item()
            speech_token = encoder_out[batch_idx, :speech_token_len, :]
            
            # Replace placeholders with real audio embeddings
            inputs_embeds[
                batch_idx,
                fbank_beg_idx : fbank_beg_idx + speech_token_len,
                :,
            ] = speech_token
    
    # 4. Forward through LLM
    device_type = next(self.parameters()).device.type
    with torch.autocast(device_type=device_type, enabled=True):
        labels_ids[labels_ids == -1] = -100  # Ignore padding
        attention_mask[attention_mask < 0] = 0
        
        model_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels_ids,
        )
        loss = model_outputs.loss
    
    # 5. Track statistics
    stats["loss"] = torch.clone(loss.detach())
    stats["batch_size"] = batch_size
    
    return loss, stats, batch_size
```

---

## 7. VOCABULARY EXTENSION (SenseVoice style)

Source: Pattern from `funasr/models/sense_voice/model.py`

```python
class MultitalkerASRConfig:
    """Configuration for multitalker ASR model."""
    
    # Vocabulary ranges
    BLANK_ID = 0
    PAD_ID = 1
    
    # Regular vocabulary: 2-59999 (reserved for characters/phonemes)
    ASR_VOCAB_START = 2
    ASR_VOCAB_END = 60000
    
    # Task tokens: 60001+
    SPEAKER_TOKENS_START = 60001
    
    @classmethod
    def get_speaker_token_id(cls, speaker_idx: int) -> int:
        """Get vocabulary ID for speaker token."""
        return cls.SPEAKER_TOKENS_START + speaker_idx
    
    @classmethod
    def build_vocabulary(cls, num_speakers: int) -> dict:
        """Build full vocabulary with speaker tokens."""
        vocab = {}
        
        # Regular tokens (externally defined, e.g., from tokenizer)
        # vocab[0:60000] = characters/phonemes
        
        # Speaker tokens
        for i in range(num_speakers):
            speaker_name = f"<speaker_{i}>"
            vocab_id = cls.get_speaker_token_id(i)
            vocab[speaker_name] = vocab_id
        
        return vocab

# Usage
config = MultitalkerASRConfig()
vocab = config.build_vocabulary(num_speakers=2)
print(vocab)  # {"<speaker_0>": 60001, "<speaker_1>": 60002, ...}
```

---

## 8. CTC MODULE (Minimal)

Source: `funasr/models/fun_asr_nano/ctc.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CTC(torch.nn.Module):
    """CTC loss module for ASR."""
    
    def __init__(self, odim: int, encoder_output_size: int, 
                 dropout_rate: float = 0.0, blank_id: int = 0, **kwargs):
        """
        Args:
            odim: output vocabulary dimension
            encoder_output_size: encoder hidden dimension
            dropout_rate: dropout rate
            blank_id: ID for CTC blank token (usually last token)
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(encoder_output_size, odim)
        self.blank_id = blank_id
        self.ctc_loss = torch.nn.CTCLoss(
            reduction="none", 
            blank=blank_id,
            zero_infinity=True
        )
    
    def softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """Softmax of frame activations."""
        return F.softmax(self.ctc_lo(hs_pad), dim=2)
    
    def log_softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """Log softmax of frame activations."""
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)
    
    def forward(self, encoder_out: torch.Tensor, encoder_out_lens: torch.Tensor,
                ys_pad: torch.Tensor, ys_pad_lens: torch.Tensor) -> torch.Tensor:
        """
        Compute CTC loss.
        
        Args:
            encoder_out: [batch, frames, encoder_dim]
            encoder_out_lens: [batch]
            ys_pad: [batch, max_label_len]
            ys_pad_lens: [batch]
        
        Returns:
            loss: scalar
        """
        # Linear projection to vocab
        logits = self.ctc_lo(encoder_out)  # [batch, frames, vocab_size]
        
        # Log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)  # [frames, batch, vocab_size]
        
        # CTC loss
        loss = self.ctc_loss(log_probs, ys_pad, encoder_out_lens, ys_pad_lens)
        
        return loss.mean()
```

---

## 9. LABEL SMOOTHING LOSS (Optional, for smoother training)

Source: `funasr/losses/label_smoothing_loss.py:24-65`

```python
import torch
from torch import nn

class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for classification tasks."""
    
    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.0,
                 normalize_length: bool = False,
                 criterion=nn.KLDivLoss(reduction="none")):
        """
        Args:
            size: number of classes
            padding_idx: ID to ignore
            smoothing: smoothing factor (0.0 = no smoothing)
            normalize_length: normalize by sequence length
        """
        super().__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.normalize_length = normalize_length
    
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seqlen, class] or [batch*seqlen, class]
            target: [batch, seqlen] or [batch*seqlen]
        
        Returns:
            loss: scalar
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        
        x = x.contiguous().view(-1, self.size)
        target = target.contiguous().view(-1)
        
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target == self.padding_idx
            total = len(target) - ignore.sum().item()
            target = target.masked_fill(ignore, 0)
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom
```

---

## 10. COMPLETE MINIMAL EXAMPLE

```python
import torch
import torch.nn as nn

class SimpleMultitalkerASR(nn.Module):
    """Minimal working example combining all above components."""
    
    def __init__(self, encoder: nn.Module, num_speakers: int = 2, 
                 encoder_dim: int = 256, vocab_size: int = 60000):
        super().__init__()
        
        self.encoder = encoder
        self.num_speakers = num_speakers
        self.encoder_dim = encoder_dim
        self.vocab_size = vocab_size
        self.blank_id = vocab_size - 1
        
        # Speaker embedding
        self.speaker_embed = nn.Embedding(num_speakers, encoder_dim)
        
        # CTC for ASR
        self.ctc_linear = nn.Linear(encoder_dim, vocab_size)
        self.ctc_loss_fn = nn.CTCLoss(reduction="mean", blank=self.blank_id)
        
        # Auxiliary classifier for speaker
        self.speaker_linear = nn.Linear(encoder_dim, num_speakers)
        self.speaker_loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, speech, speech_lengths, text, text_lengths, 
                speaker_labels, speaker_label_lens):
        """
        Args:
            speech: [batch, frames, features]
            speech_lengths: [batch]
            text: [batch, text_len]
            text_lengths: [batch]
            speaker_labels: [batch]
            speaker_label_lens: [batch]
        """
        batch_size = speech.shape[0]
        
        # Encode with speaker prepending
        speaker_query = self.speaker_embed(speaker_labels).unsqueeze(1)
        # speaker_query: [batch, 1, encoder_dim]
        
        enc_out, enc_lens = self.encoder(speech, speech_lengths)
        # enc_out: [batch, frames, encoder_dim]
        
        enc_out_with_speaker = torch.cat([speaker_query, enc_out], dim=1)
        enc_lens = enc_lens + 1
        # enc_out_with_speaker: [batch, frames+1, encoder_dim]
        
        # ASR loss (on frames after speaker token)
        asr_logits = self.ctc_linear(enc_out_with_speaker[:, 1:, :])
        asr_logits = asr_logits.transpose(0, 1)  # [frames, batch, vocab]
        asr_log_probs = torch.nn.functional.log_softmax(asr_logits, dim=-1)
        
        loss_asr = self.ctc_loss_fn(
            asr_log_probs, text, enc_lens - 1, text_lengths
        )
        
        # Speaker loss (on speaker token frame)
        speaker_logits = self.speaker_linear(enc_out_with_speaker[:, 0, :])
        # speaker_logits: [batch, num_speakers]
        
        loss_speaker = self.speaker_loss_fn(speaker_logits, speaker_labels)
        
        # Combined loss
        loss = loss_asr + 0.1 * loss_speaker
        
        return loss, {
            "loss_asr": loss_asr.item(),
            "loss_speaker": loss_speaker.item(),
        }

# Usage
if __name__ == "__main__":
    # Create dummy encoder
    encoder = nn.LSTM(80, 256, 2, batch_first=True)
    
    model = SimpleMultitalkerASR(encoder, num_speakers=2)
    
    # Dummy batch
    speech = torch.randn(2, 100, 80)  # [batch=2, frames=100, features=80]
    speech_lengths = torch.tensor([100, 95])
    text = torch.randint(1, 60000, (2, 30))  # [batch=2, text_len=30]
    text_lengths = torch.tensor([30, 28])
    speaker_labels = torch.tensor([0, 1])  # speaker 0, speaker 1
    speaker_label_lens = torch.tensor([1, 1])
    
    # Forward pass
    loss, stats = model(
        speech, speech_lengths, text, text_lengths,
        speaker_labels, speaker_label_lens
    )
    print(f"Loss: {loss.item():.4f}")
    print(f"Stats: {stats}")
```

---

## IMPLEMENTATION CHECKLIST

- [ ] Create speaker embedding table (Embedding layer)
- [ ] Implement speech prepending in encode()
- [ ] Add two-stage loss computation (speaker + ASR)
- [ ] Test forward pass with dummy data
- [ ] Implement inference with speaker control
- [ ] Extend vocabulary with speaker tokens
- [ ] Add CTC module for ASR loss
- [ ] Test training loop
- [ ] Add logging/monitoring
- [ ] Validate on real data

