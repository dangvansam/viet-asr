#!/bin/bash
# Full Multi-Task Training Pipeline for Vietnamese ASR
# 3-Phase Curriculum: ASR → Multi-talker → Paralinguistic
#
# Usage:
#   bash scripts/run_multitask_training.sh [phase]
#   phase: 1 (ASR) | 2 (multitalker) | 3 (paralinguistic) | all (default)

set -e

export CUDA_VISIBLE_DEVICES=1

# ============== Configuration ==============
DATA_ROOT="/home/samdv/DATA/tts"
OUTPUT_DIR="data"
CHECKPOINT_DIR="checkpoints/multitask"
BASE_MODEL="models/multitalker-parakeet-streaming-0.6b-v1.nemo"
VOCAB_SIZE=2048

# Common training params
BATCH_SIZE=16
GRAD_ACCUM=4
MAX_SPEAKERS=4
WANDB_PROJECT="multitask-asr"

# Phase to run (default: all)
PHASE="${1:-all}"

echo "=============================================="
echo "Multi-Task Vietnamese ASR Training Pipeline"
echo "=============================================="
echo "Phase: $PHASE"
echo ""

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"

# ============== Step 1: Prepare base manifests ==============
if [ ! -f "$OUTPUT_DIR/train_single_speaker.json" ]; then
    echo "[Step 1] Preparing single-speaker data manifests..."
    uv run scripts/prepare_all_tts_data.py \
        --data_root "$DATA_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --val_split 0.05
else
    echo "[Step 1] Single-speaker manifests exist. Skipping..."
fi

# ============== Step 2: Generate speaker metadata template ==============
if [ ! -f "$OUTPUT_DIR/speaker_metadata.json" ]; then
    echo ""
    echo "[Step 2] Generating speaker metadata template..."
    uv run scripts/prepare_multitask_data.py \
        --input_manifest "$OUTPUT_DIR/train_single_speaker.json" \
        --output_manifest "$OUTPUT_DIR/train_multitask.json" \
        --generate_template \
        --speaker_metadata "$OUTPUT_DIR/speaker_metadata.json"
    echo ""
    echo ">>> IMPORTANT: Review and edit $OUTPUT_DIR/speaker_metadata.json"
    echo ">>> Correct gender/age labels for each speaker, then re-run this script."
    echo ""
fi

# ============== Step 3: Annotate manifests with task labels ==============
if [ ! -f "$OUTPUT_DIR/train_multitask.json" ]; then
    echo "[Step 3] Annotating manifests with multi-task labels..."
    uv run scripts/prepare_multitask_data.py \
        --input_manifest "$OUTPUT_DIR/train_single_speaker.json" \
        --output_manifest "$OUTPUT_DIR/train_multitask.json" \
        --speaker_metadata "$OUTPUT_DIR/speaker_metadata.json" \
        --val_manifest "$OUTPUT_DIR/val_single_speaker.json" \
        --val_output "$OUTPUT_DIR/val_multitask.json"
else
    echo "[Step 3] Multi-task manifests exist. Skipping..."
fi

# ============== Step 4: Tokenizer (reuse if exists) ==============
if [ ! -f "$OUTPUT_DIR/vi_tokenizer.model" ]; then
    echo ""
    echo "[Step 4] Preparing tokenizer..."

    if [ ! -f "$OUTPUT_DIR/vietnamese_corpus.txt" ]; then
        uv run scripts/prepare_tokenizer_corpus.py \
            --manifests "$OUTPUT_DIR/train_single_speaker.json" \
            --output "$OUTPUT_DIR/vietnamese_corpus.txt"
    fi

    uv run scripts/train_tokenizer.py \
        --input "$OUTPUT_DIR/vietnamese_corpus.txt" \
        --model_prefix "$OUTPUT_DIR/vi_tokenizer" \
        --vocab_size "$VOCAB_SIZE"
else
    echo "[Step 4] Tokenizer exists. Skipping..."
fi

# ============== Phase 1: Vietnamese ASR Fine-tuning ==============
run_phase1() {
    echo ""
    echo "=============================================="
    echo "Phase 1: Vietnamese ASR Fine-tuning"
    echo "  - Single-speaker data"
    echo "  - Freeze lower 18 encoder layers"
    echo "  - RNNT loss only"
    echo "=============================================="

    uv run scripts/train_multitask.py \
        --mode finetune \
        --curriculum_phase asr \
        --model_path "$BASE_MODEL" \
        --tokenizer_dir "$OUTPUT_DIR/vi_tokenizer.model" \
        --vocab_size "$VOCAB_SIZE" \
        --train_manifest "$OUTPUT_DIR/train_single_speaker.json" \
        --val_manifest "$OUTPUT_DIR/val_single_speaker.json" \
        --max_epochs 20 \
        --batch_size "$BATCH_SIZE" \
        --learning_rate 1e-4 \
        --accumulate_grad_batches "$GRAD_ACCUM" \
        --num_frozen_layers 18 \
        --checkpoint_dir "$CHECKPOINT_DIR/phase1_asr" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "phase1_asr"

    echo "Phase 1 complete. Checkpoint at: $CHECKPOINT_DIR/phase1_asr/"
}

# ============== Phase 2: Multi-talker Adaptation ==============
run_phase2() {
    echo ""
    echo "=============================================="
    echo "Phase 2: Multi-talker Adaptation"
    echo "  - On-the-fly multi-speaker synthesis"
    echo "  - Freeze lower 12 encoder layers"
    echo "  - RNNT + Sortformer loss"
    echo "=============================================="

    # Find best Phase 1 checkpoint
    PHASE1_CKPT=$(find "$CHECKPOINT_DIR/phase1_asr" -name "*.ckpt" -path "*/checkpoints/*" | sort | tail -1)
    if [ -z "$PHASE1_CKPT" ]; then
        echo "ERROR: No Phase 1 checkpoint found. Run Phase 1 first."
        exit 1
    fi
    echo "Resuming from: $PHASE1_CKPT"

    uv run scripts/train_multitask.py \
        --mode finetune \
        --curriculum_phase multitalker \
        --model_path "$PHASE1_CKPT" \
        --train_manifest "$OUTPUT_DIR/train_single_speaker.json" \
        --val_manifest "$OUTPUT_DIR/val_single_speaker.json" \
        --max_epochs 15 \
        --batch_size "$BATCH_SIZE" \
        --learning_rate 1e-5 \
        --accumulate_grad_batches "$GRAD_ACCUM" \
        --max_speakers "$MAX_SPEAKERS" \
        --num_frozen_layers 12 \
        --checkpoint_dir "$CHECKPOINT_DIR/phase2_multitalker" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "phase2_multitalker"

    echo "Phase 2 complete. Checkpoint at: $CHECKPOINT_DIR/phase2_multitalker/"
}

# ============== Phase 3: Paralinguistic SFT ==============
run_phase3() {
    echo ""
    echo "=============================================="
    echo "Phase 3: Paralinguistic SFT"
    echo "  - Multi-task annotated data"
    echo "  - Prompt tokens active (emotion/gender/age/voice_state)"
    echo "  - Dynamic loss weighting"
    echo "  - Low learning rate"
    echo "=============================================="

    # Find best Phase 2 checkpoint
    PHASE2_CKPT=$(find "$CHECKPOINT_DIR/phase2_multitalker" -name "*.ckpt" -path "*/checkpoints/*" | sort | tail -1)
    if [ -z "$PHASE2_CKPT" ]; then
        echo "ERROR: No Phase 2 checkpoint found. Run Phase 2 first."
        exit 1
    fi
    echo "Resuming from: $PHASE2_CKPT"

    uv run scripts/train_multitask.py \
        --mode finetune \
        --curriculum_phase paralinguistic \
        --model_path "$PHASE2_CKPT" \
        --train_manifest "$OUTPUT_DIR/train_multitask.json" \
        --val_manifest "$OUTPUT_DIR/val_multitask.json" \
        --multitask_train_manifest "$OUTPUT_DIR/train_multitask.json" \
        --max_epochs 10 \
        --batch_size "$BATCH_SIZE" \
        --learning_rate 5e-6 \
        --accumulate_grad_batches "$GRAD_ACCUM" \
        --max_speakers "$MAX_SPEAKERS" \
        --num_frozen_layers 0 \
        --use_dynamic_weighting \
        --prompt_ce_weight 1.0 \
        --checkpoint_dir "$CHECKPOINT_DIR/phase3_paralinguistic" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "phase3_paralinguistic"

    echo "Phase 3 complete. Checkpoint at: $CHECKPOINT_DIR/phase3_paralinguistic/"
}

# ============== Run selected phase(s) ==============
case "$PHASE" in
    1)     run_phase1 ;;
    2)     run_phase2 ;;
    3)     run_phase3 ;;
    all)
        run_phase1
        run_phase2
        run_phase3
        ;;
    *)
        echo "Usage: $0 [1|2|3|all]"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Training Pipeline Complete!"
echo "=============================================="
echo ""
echo "To run inference:"
echo "  uv run scripts/infer_multitask.py \\"
echo "    --audio_path test.wav \\"
echo "    --model_path $CHECKPOINT_DIR/phase3_paralinguistic/last.ckpt \\"
echo "    --use_post_processor"
