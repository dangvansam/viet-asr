# Training Vietnamese Multitalker ASR Model from Scratch

This guide covers **three approaches** to train a Vietnamese multitalker ASR model, ranging from easiest (leveraging pretrained weights) to most challenging (full scratch training).

## Approach Comparison

| Approach | Training Time | Data Required | Final Model | Complexity |
|----------|--------------|---------------|-------------|------------|
| **1. Extend Tokenizer** | 12-24 hours | 100+ hours | Bilingual (Vi+En) | ⭐ Easy |
| **2. Replace Tokenizer** | 24-48 hours | 200+ hours | Vietnamese-only | ⭐⭐ Medium |
| **3. Full Scratch** | 7-14 days | 500+ hours | Vietnamese-only | ⭐⭐⭐⭐⭐ Very Hard |

---

## Approach 1: Extend Tokenizer (Recommended)

**Best for:** Users who want Vietnamese + English + Vietlish support with fastest training.

**How it works:**
- Merges English (1024 tokens) + Vietnamese (1024 tokens) = ~2048 total
- Preserves pretrained English embeddings
- Adds new Vietnamese token embeddings
- Fine-tunes on Vietnamese data

**Advantages:**
- ✅ Fastest training (10K-20K steps)
- ✅ Supports code-switching (Vietlish)
- ✅ Maintains English capability
- ✅ Leverages all pretrained weights

**Workflow:**

```bash
# Step 1: Prepare Vietnamese corpus
python scripts/prepare_tokenizer_corpus.py \
    --manifests data/train.json data/train_mixed.json \
    --output data/vietnamese_corpus.txt

# Step 2: Train Vietnamese tokenizer
python scripts/train_tokenizer.py \
    --input data/vietnamese_corpus.txt \
    --model_prefix data/vi_tokenizer \
    --vocab_size 1024

# Step 3: Merge English + Vietnamese vocabularies
python scripts/merge_tokenizers.py \
    --english_model checkpoints/pretrained/multitalker-parakeet-streaming-0.6b-v1.nemo \
    --vietnamese_model data/vi_tokenizer.model \
    --output_vocab data/merged_vocab.txt \
    --output_mapping data/token_mapping.json

# Step 4: Retrain SentencePiece with merged vocabulary
spm_train --input=data/vietnamese_corpus.txt \
          --model_prefix=data/merged_tokenizer \
          --vocab_size=$(jq -r '.new_vocab_size' data/token_mapping.json) \
          --vocabulary=data/merged_vocab.txt \
          --model_type=bpe \
          --character_coverage=1.0 \
          --normalization_rule_name=identity

# Step 5: Extend model with merged tokenizer
python scripts/extend_tokenizer.py \
    --base_model checkpoints/pretrained/multitalker-parakeet-streaming-0.6b-v1.nemo \
    --merged_vocab data/merged_vocab.txt \
    --token_mapping data/token_mapping.json \
    --merged_tokenizer data/merged_tokenizer.model \
    --output_model checkpoints/multitalker-vietnamese-extended.nemo

# Step 6: Verify
python scripts/verify_tokenizer.py \
    --model_path checkpoints/multitalker-vietnamese-extended.nemo \
    --vocab_size $(($(jq -r '.new_vocab_size' data/token_mapping.json) + 1))

# Step 7: Fine-tune on Vietnamese data
python scripts/finetune.py \
    --model_path checkpoints/multitalker-vietnamese-extended.nemo \
    --train_manifest data/train_mixed.json \
    --val_manifest data/val_mixed.json \
    --gpus 1 \
    --max_steps 10000 \
    --learning_rate 1e-5
```

**Expected Results:**
- After 5,000 steps: WER ~25-35% on Vietnamese
- After 10,000 steps: WER ~15-25% on Vietnamese
- Maintains English transcription
- Handles Vietlish naturally

---

## Approach 2: Replace Tokenizer (Vietnamese-Only)

**Best for:** Users who only need Vietnamese support and want faster convergence than full scratch.

**How it works:**
- Trains Vietnamese-only tokenizer (2048 tokens)
- Replaces English tokenizer completely
- Randomizes decoder/joint layer weights
- Preserves encoder weights (acoustic features are language-agnostic)
- Fine-tunes on Vietnamese data

**Advantages:**
- ✅ Pure Vietnamese model (smaller vocab, faster inference)
- ✅ Leverages pretrained encoder (acoustic feature extraction)
- ✅ Faster than full scratch training
- ✅ No English "interference"

**Disadvantages:**
- ❌ Cannot transcribe English
- ❌ Cannot handle code-switching
- ❌ Requires more training than Approach 1

**Workflow:**

```bash
# Step 1: Prepare Vietnamese corpus
python scripts/prepare_tokenizer_corpus.py \
    --manifests data/train.json data/train_mixed.json \
    --output data/vietnamese_corpus.txt

# Step 2: Train Vietnamese-only tokenizer (larger vocab)
python scripts/train_tokenizer.py \
    --input data/vietnamese_corpus.txt \
    --model_prefix data/vi_tokenizer \
    --vocab_size 2048

# Step 3: Replace tokenizer in pretrained model
python scripts/train_from_scratch.py \
    --mode replace \
    --base_model checkpoints/pretrained/multitalker-parakeet-streaming-0.6b-v1.nemo \
    --vietnamese_tokenizer data/vi_tokenizer.model \
    --vocab_size 2048 \
    --output_model checkpoints/multitalker-vietnamese-only.nemo

# Step 4: Verify
python scripts/verify_tokenizer.py \
    --model_path checkpoints/multitalker-vietnamese-only.nemo \
    --vocab_size 2049  # 2048 + 1 blank

# Step 5: Fine-tune with higher learning rate initially (2-stage training)

# Stage 1: Decoder/Joint warmup (steps 0-2000)
python scripts/finetune.py \
    --model_path checkpoints/multitalker-vietnamese-only.nemo \
    --train_manifest data/train_mixed.json \
    --val_manifest data/val_mixed.json \
    --gpus 1 \
    --max_steps 2000 \
    --learning_rate 5e-5 \
    --batch_size 8

# Stage 2: Full fine-tuning (steps 2000-20000)
python scripts/finetune.py \
    --model_path checkpoints/multitalker-vietnamese-only-finetuned.nemo \
    --train_manifest data/train_mixed.json \
    --val_manifest data/val_mixed.json \
    --gpus 1 \
    --max_steps 18000 \
    --learning_rate 1e-5 \
    --batch_size 4
```

**Training Configuration:**

| Parameter | Stage 1 (Warmup) | Stage 2 (Full) |
|-----------|------------------|----------------|
| Steps | 0-2000 | 2000-20000 |
| Learning Rate | 5e-5 | 1e-5 |
| Batch Size | 8 | 4 |
| Gradient Accumulation | 2 | 4 |
| Focus | Decoder/Joint | All layers |

**Expected Results:**
- After 5,000 steps: WER ~35-45% on Vietnamese
- After 10,000 steps: WER ~25-35% on Vietnamese
- After 20,000 steps: WER ~15-25% on Vietnamese
- No English capability

---

## Approach 3: Full Scratch Training (Advanced)

**Best for:** Research purposes or when you have massive datasets (500+ hours) and want complete control.

**How it works:**
- Initialize model from scratch with NeMo config
- Train all weights from random initialization
- Requires custom NeMo YAML configuration
- Needs very large datasets and long training

**Advantages:**
- ✅ Complete control over architecture
- ✅ No pretrained biases
- ✅ Can customize model size, layers, etc.

**Disadvantages:**
- ❌ Requires 500+ hours of high-quality data
- ❌ Training takes 7-14 days on GPU
- ❌ Needs deep NeMo expertise
- ❌ High risk of training instability
- ❌ Much more expensive (compute costs)

**Workflow:**

### Step 1: Create NeMo Configuration

Create `configs/vietnamese_multitalker.yaml`:

```yaml
name: MultitalkerVietnamese

model:
  # Sample rate
  sample_rate: 16000

  # Tokenizer
  tokenizer:
    dir: /tokenizers/vietnamese
    type: bpe
    vocab_size: 2048

  # Preprocessor
  preprocessor:
    _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
    sample_rate: 16000
    window_size: 0.025
    window_stride: 0.01
    n_window_size: 400
    n_window_stride: 160
    features: 128
    n_fft: 512
    window: hann

  # Encoder (Conformer)
  encoder:
    _target_: nemo.collections.asr.modules.ConformerEncoder
    feat_in: 128
    n_layers: 24
    d_model: 1024
    feat_out: 1024
    subsampling: dw_striding
    subsampling_factor: 8
    ff_expansion_factor: 4
    self_attention_model: rel_pos
    n_heads: 8
    att_context_size: [-1, -1]
    xscaling: true
    pos_emb_max_len: 5000

  # Decoder (RNNT)
  decoder:
    _target_: nemo.collections.asr.modules.RNNTDecoder
    prednet:
      pred_hidden: 640
      pred_rnn_layers: 2
      dropout: 0.2
    vocab_size: 2049  # 2048 + 1 blank

  # Joint Network
  joint:
    _target_: nemo.collections.asr.modules.RNNTJoint
    jointnet:
      encoder_hidden: 1024
      pred_hidden: 640
      joint_hidden: 640
      activation: relu
      dropout: 0.2
    num_classes: 2049

  # Training data
  train_ds:
    manifest_filepath: ???
    sample_rate: 16000
    batch_size: 16
    num_workers: 8
    max_duration: 20.0
    min_duration: 0.1

  # Validation data
  validation_ds:
    manifest_filepath: ???
    sample_rate: 16000
    batch_size: 8
    num_workers: 4

  # Optimizer
  optim:
    name: adamw
    lr: 0.001
    betas: [0.9, 0.98]
    weight_decay: 1e-3
    sched:
      name: NoamAnnealing
      d_model: 1024
      warmup_steps: 10000
      min_lr: 1e-6

trainer:
  devices: 4  # Number of GPUs
  num_nodes: 1
  max_steps: 200000
  val_check_interval: 1000
  precision: 16
  accelerator: gpu
  strategy: ddp
```

### Step 2: Train Model from Scratch

**Using NeMo's native training:**

```bash
# Install NeMo
pip install nemo_toolkit[asr]

# Prepare Vietnamese tokenizer
python scripts/train_tokenizer.py \
    --input data/vietnamese_corpus.txt \
    --model_prefix data/vi_tokenizer \
    --vocab_size 2048

# Train model (requires multi-GPU setup)
python -m nemo.collections.asr.models.rnnt_bpe_model \
    --config-path=configs \
    --config-name=vietnamese_multitalker \
    model.train_ds.manifest_filepath=data/train_mixed.json \
    model.validation_ds.manifest_filepath=data/val_mixed.json \
    model.tokenizer.dir=data/vi_tokenizer \
    trainer.devices=4 \
    trainer.max_steps=200000 \
    exp_manager.exp_dir=checkpoints/scratch_training \
    exp_manager.name=vietnamese_multitalker
```

**Training Requirements:**
- **Data:** 500+ hours of clean Vietnamese audio
- **GPUs:** 4x A100 or 8x V100 (minimum)
- **Time:** 7-14 days continuous training
- **Disk:** 500GB+ for checkpoints
- **Cost:** $5,000-$15,000 in cloud GPU costs

**Training Monitoring:**

```bash
# Monitor with TensorBoard
tensorboard --logdir checkpoints/scratch_training

# Key metrics to watch:
# - Training loss: Should decrease steadily
# - Validation WER: Target <30% by 100K steps
# - Learning rate: NoamAnnealing schedule
# - Gradient norms: Watch for instability
```

**Expected Results:**
- After 50,000 steps: WER ~50-60%
- After 100,000 steps: WER ~30-40%
- After 200,000 steps: WER ~20-30%
- Converges slower than transfer learning

---

## Comparison Summary

### Data Requirements

| Approach | Minimum | Recommended | Optimal |
|----------|---------|-------------|---------|
| Extend | 50 hours | 100 hours | 200+ hours |
| Replace | 100 hours | 200 hours | 500+ hours |
| Full Scratch | 300 hours | 500 hours | 1000+ hours |

### Training Time (on single V100)

| Approach | Steps | Time | GPU Hours |
|----------|-------|------|-----------|
| Extend | 10,000 | 12-24 hours | 12-24 |
| Replace | 20,000 | 24-48 hours | 24-48 |
| Full Scratch | 200,000 | 7-14 days | 168-336 |

### Final Model Performance (on clean Vietnamese)

| Approach | Expected WER | English Support | Vietlish |
|----------|--------------|-----------------|----------|
| Extend | 15-25% | ✅ Yes | ✅ Yes |
| Replace | 15-25% | ❌ No | ❌ No |
| Full Scratch | 20-30% | ❌ No | ❌ No |

---

## Recommendation Matrix

**Choose Extend if:**
- ✅ You have 100+ hours of Vietnamese data
- ✅ You need English + Vietnamese support
- ✅ You need code-switching (Vietlish)
- ✅ You want fastest training time
- ✅ You have limited GPU resources

**Choose Replace if:**
- ✅ You have 200+ hours of Vietnamese data
- ✅ You only need Vietnamese (no English)
- ✅ You want a pure Vietnamese model
- ✅ You have moderate GPU resources
- ❌ You don't need code-switching

**Choose Full Scratch if:**
- ✅ You have 500+ hours of high-quality data
- ✅ You need complete architecture control
- ✅ You have multi-GPU cluster (4+ GPUs)
- ✅ You have research budget ($5K-$15K)
- ✅ You have deep NeMo expertise
- ❌ Transfer learning doesn't meet your needs

---

## Quick Start Guide

### For Most Users (Extend Approach)

```bash
# 1. Prepare data
python scripts/prepare_tokenizer_corpus.py \
    --manifests data/train.json \
    --output data/vietnamese_corpus.txt

# 2. Train Vietnamese tokenizer
python scripts/train_tokenizer.py \
    --input data/vietnamese_corpus.txt \
    --model_prefix data/vi_tokenizer \
    --vocab_size 1024

# 3. Merge vocabularies
python scripts/merge_tokenizers.py \
    --english_model checkpoints/pretrained/multitalker-parakeet-streaming-0.6b-v1.nemo \
    --vietnamese_model data/vi_tokenizer.model \
    --output_vocab data/merged_vocab.txt \
    --output_mapping data/token_mapping.json

# 4. Retrain merged tokenizer
spm_train --input=data/vietnamese_corpus.txt \
          --model_prefix=data/merged_tokenizer \
          --vocab_size=$(jq -r '.new_vocab_size' data/token_mapping.json) \
          --vocabulary=data/merged_vocab.txt \
          --model_type=bpe \
          --character_coverage=1.0 \
          --normalization_rule_name=identity

# 5. Extend model
python scripts/extend_tokenizer.py \
    --base_model checkpoints/pretrained/multitalker-parakeet-streaming-0.6b-v1.nemo \
    --merged_vocab data/merged_vocab.txt \
    --token_mapping data/token_mapping.json \
    --merged_tokenizer data/merged_tokenizer.model \
    --output_model checkpoints/multitalker-vietnamese.nemo

# 6. Fine-tune
python scripts/finetune.py \
    --model_path checkpoints/multitalker-vietnamese.nemo \
    --train_manifest data/train.json \
    --val_manifest data/val.json \
    --gpus 1 \
    --max_steps 10000
```

### For Vietnamese-Only (Replace Approach)

```bash
# 1-2. Same as above

# 3. Train Vietnamese-only tokenizer (larger vocab)
python scripts/train_tokenizer.py \
    --input data/vietnamese_corpus.txt \
    --model_prefix data/vi_tokenizer \
    --vocab_size 2048

# 4. Replace tokenizer
python scripts/train_from_scratch.py \
    --mode replace \
    --base_model checkpoints/pretrained/multitalker-parakeet-streaming-0.6b-v1.nemo \
    --vietnamese_tokenizer data/vi_tokenizer.model \
    --vocab_size 2048 \
    --output_model checkpoints/multitalker-vietnamese-only.nemo

# 5. Fine-tune (2-stage)
# Stage 1: Warmup
python scripts/finetune.py \
    --model_path checkpoints/multitalker-vietnamese-only.nemo \
    --train_manifest data/train.json \
    --val_manifest data/val.json \
    --gpus 1 \
    --max_steps 2000 \
    --learning_rate 5e-5

# Stage 2: Full training
python scripts/finetune.py \
    --model_path checkpoints/multitalker-vietnamese-only-finetuned.nemo \
    --train_manifest data/train.json \
    --val_manifest data/val.json \
    --gpus 1 \
    --max_steps 18000 \
    --learning_rate 1e-5
```

---

## Troubleshooting

### Issue: Training loss not decreasing

**Solution:**
1. Check tokenizer - verify low UNK rate on Vietnamese text
2. Reduce learning rate by 2-5x
3. Increase batch size or gradient accumulation
4. Verify data quality (clean audio, accurate transcripts)

### Issue: Out of memory during training

**Solution:**
1. Reduce batch size: `--batch_size 2`
2. Increase gradient accumulation: `--accumulate_grad_batches 8`
3. Use mixed precision: `--precision 16`
4. Reduce audio max duration in data config

### Issue: Model overfits quickly

**Solution:**
1. Add more training data
2. Increase dropout in decoder config
3. Add data augmentation (speed, noise)
4. Reduce model complexity (fewer layers)

### Issue: Poor performance on Vietnamese

**Solution:**
1. Ensure sufficient Vietnamese training data (200+ hours)
2. Check tokenizer coverage - increase vocab_size
3. Train longer (20K+ steps)
4. Verify audio quality (16kHz, clean)

---

## Best Practices

1. **Always verify tokenizer first:**
   ```bash
   python scripts/verify_tokenizer.py --model_path <model> --vocab_size <size>
   ```

2. **Monitor training closely:**
   - Watch validation WER every 1000 steps
   - Check for gradient explosions
   - Verify loss decreasing smoothly

3. **Use checkpointing:**
   - Save every 1000 steps
   - Keep best 3 checkpoints
   - Test on validation set regularly

4. **Start small, scale up:**
   - Test with 10% of data first
   - Verify pipeline works end-to-end
   - Then scale to full dataset

5. **Staged fine-tuning:**
   - Stage 1: Higher LR, decoder focus
   - Stage 2: Lower LR, full model
   - Prevents catastrophic forgetting

---

## Additional Resources

- **NeMo Documentation:** https://docs.nvidia.com/nemo-framework/user-guide/latest/
- **ASR Training Tutorial:** https://github.com/NVIDIA/NeMo/tree/main/tutorials/asr
- **Multitalker Paper:** https://arxiv.org/abs/2310.07313
- **SentencePiece Guide:** https://github.com/google/sentencepiece

---

## Support

For issues or questions:
1. Check this documentation
2. Verify tokenizer with verification script
3. Review training logs for errors
4. Check NeMo GitHub issues: https://github.com/NVIDIA/NeMo/issues
