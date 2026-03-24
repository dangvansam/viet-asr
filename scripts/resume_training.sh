#!/bin/bash
# Resume training from a PyTorch Lightning checkpoint

set -e

export CUDA_VISIBLE_DEVICES=1

# Required input
CHECKPOINT_PATH="$1"

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Usage: ./scripts/resume_training.sh <path_to_checkpoint.ckpt>"
    echo "Example: ./scripts/resume_training.sh multitalker-asr-v2/i5s6qvc5/checkpoints/epoch=4-step=19590.ckpt"
    exit 1
fi

echo "=============================================="
echo "Resuming Training from: $CHECKPOINT_PATH"
echo "=============================================="

OUTPUT_DIR="data"
CHECKPOINT_DIR="checkpoints"
CONFIG_MODEL="model_config.yaml"
VOCAB_SIZE=2048

# Training parameters
MAX_STEPS=500000
BATCH_SIZE=64
LEARNING_RATE=0.5
MAX_SPEAKERS=4
GRAD_ACCUM=4

uv run scripts/finetune.py \
    --model_path "$CHECKPOINT_PATH" \
    --config_path "$CONFIG_MODEL" \
    --tokenizer_dir "$OUTPUT_DIR/vi_tokenizer.model" \
    --vocab_size "$VOCAB_SIZE" \
    --output_path "$CHECKPOINT_DIR/multitalker-vietnamese-resumed.nemo" \
    --train_manifest "$OUTPUT_DIR/train_single_speaker.json" \
    --val_manifest "$OUTPUT_DIR/val_single_speaker.json" \
    --max_steps "$MAX_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --accumulate_grad_batches "$GRAD_ACCUM" \
    --gpus 1 \
    --wandb_project "multitalker-asr-v2" \
    --use_on_the_fly_synthesis \
    --max_speakers "$MAX_SPEAKERS" \
    --synthesis_num_workers 8
