#!/bin/bash
# Full Training Pipeline for Vietnamese Multitalker ASR
# This script trains the model from scratch using all TTS datasets

set -e  # Exit on error

export CUDA_VISIBLE_DEVICES=1

# Configuration
DATA_ROOT="/home/samdv/DATA/tts"
OUTPUT_DIR="data"
CHECKPOINT_DIR="checkpoints"
BASE_MODEL="models/multitalker-parakeet-streaming-0.6b-v1.nemo"
VOCAB_SIZE=2048  # Vietnamese vocabulary size

# Training parameters
MAX_STEPS=500000  # Increased for better convergence when training from scratch
BATCH_SIZE=32
LEARNING_RATE=0.5  # Lower LR for training from scratch (Noam scheduler will scale this)
NUM_SYNTH_SAMPLES=1000000  # Number of multi-speaker samples to synthesize
MAX_SPEAKERS=2  # Start with 2 speakers for better scratch convergence
GRAD_ACCUM=8  # Gradient accumulation for effective batch size of 256

echo "=============================================="
echo "Vietnamese Multitalker ASR Training Pipeline"
echo "=============================================="
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$OUTPUT_DIR/synthesized_audio"

# Step 1: Prepare single-speaker manifests (if not already done)
if [ ! -f "$OUTPUT_DIR/train_single_speaker.json" ]; then
    echo "[Step 1/7] Preparing single-speaker data manifests..."
    uv run scripts/prepare_all_tts_data.py \
        --data_root "$DATA_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --val_split 0.05
else
    echo "[Step 1/7] Single-speaker manifests already exist. Skipping..."
fi

# Step 2: Extract text corpus for tokenizer training
echo ""
echo "[Step 2/7] Extracting text corpus for tokenizer training..."
if [ ! -f "$OUTPUT_DIR/vietnamese_corpus.txt" ]; then
    uv run scripts/prepare_tokenizer_corpus.py \
        --manifests "$OUTPUT_DIR/train_single_speaker.json" \
        --output "$OUTPUT_DIR/vietnamese_corpus.txt"
else
    echo "Corpus already exists. Skipping..."
fi

# Step 3: Train Vietnamese tokenizer
echo ""
echo "[Step 3/7] Training Vietnamese SentencePiece tokenizer..."
if [ ! -f "$OUTPUT_DIR/vi_tokenizer.model" ]; then
    uv run scripts/train_tokenizer.py \
        --input "$OUTPUT_DIR/vietnamese_corpus.txt" \
        --model_prefix "$OUTPUT_DIR/vi_tokenizer" \
        --vocab_size "$VOCAB_SIZE"
else
    echo "Tokenizer already exists. Skipping..."
fi


# Step 5: Synthesize multi-speaker training data
echo ""
echo "[Step 5/7] Synthesizing multi-speaker overlapping audio..."
if [ ! -f "$OUTPUT_DIR/train_mixed.json" ]; then
    uv run scripts/synthesize_data.py \
        --input_manifests "$OUTPUT_DIR/train_single_speaker.json" \
        --output_dir "$OUTPUT_DIR/synthesized_audio" \
        --output_manifest "$OUTPUT_DIR/train_mixed.json" \
        --num_samples "$NUM_SYNTH_SAMPLES" \
        --max_speakers "$MAX_SPEAKERS"

    # Create validation mixed data
    uv run scripts/synthesize_data.py \
        --input_manifests "$OUTPUT_DIR/val_single_speaker.json" \
        --output_dir "$OUTPUT_DIR/synthesized_audio_val" \
        --output_manifest "$OUTPUT_DIR/val_mixed.json" \
        --num_samples 5000 \
        --max_speakers "$MAX_SPEAKERS"
else
    echo "Mixed data already exists. Skipping..."
fi

# Step 6: Fine-tune the model

OUTPUT_MODEL="$CHECKPOINT_DIR/multitalker-vietnamese-scratch.nemo"
CONFIG_MODEL="model_config.yaml"

echo ""
echo "[Step 6/7] Starting training..."
uv run scripts/finetune.py \
    --model_path "$OUTPUT_MODEL" \
    --config_path "$CONFIG_MODEL" \
    --tokenizer_dir "$OUTPUT_DIR/vi_tokenizer.model" \
    --vocab_size "$VOCAB_SIZE" \
    --output_path "$OUTPUT_MODEL" \
    --train_manifest "$OUTPUT_DIR/train_mixed.json" \
    --val_manifest "$OUTPUT_DIR/val_mixed.json" \
    --max_steps "$MAX_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --accumulate_grad_batches "$GRAD_ACCUM" \
    --gpus 1 \
    --wandb_project "multitalker-asr-v2"

# Step 7: Evaluate the model
echo ""
echo "[Step 7/7] Training complete!"
echo ""
echo "Final model saved to: $CHECKPOINT_DIR/lightning_logs/"
echo ""
echo "To test the model, run:"
echo "  uv run scripts/infer.py --model_path <path_to_final_model.nemo> --audio <test_audio.wav>"

echo ""
echo "=============================================="
echo "Training Pipeline Complete!"
echo "=============================================="
