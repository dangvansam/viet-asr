#!/usr/bin/env python3
"""
Train Vietnamese-only multitalker ASR model from scratch.

This script provides two modes:
1. Replace tokenizer: Replace English tokenizer with Vietnamese, resize layers, then fine-tune
2. Full scratch: Initialize a new model architecture with Vietnamese tokenizer

Mode 1 (Replace - Recommended): Leverages pretrained encoder/decoder weights
Mode 2 (Full scratch): Requires much more data and training time

Usage:
    # Mode 1: Replace tokenizer (recommended)
    python scripts/train_from_scratch.py \
        --mode replace \
        --base_model checkpoints/pretrained/multitalker-parakeet-streaming-0.6b-v1.nemo \
        --vietnamese_tokenizer data/vi_tokenizer.model \
        --output_model checkpoints/multitalker-vietnamese-only-0.6b-v1.nemo

    # Mode 2: Full scratch (requires custom NeMo config)
    python scripts/train_from_scratch.py \
        --mode full_scratch \
        --config configs/vietnamese_multitalker.yaml \
        --vietnamese_tokenizer data/vi_tokenizer.model \
        --output_model checkpoints/multitalker-vietnamese-scratch.nemo
"""

import json
from omegaconf import open_dict
from loguru import logger
import torch.nn as nn
import torch
from nemo.collections.asr.models import ASRModel
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def replace_tokenizer_mode(base_model_path, vietnamese_tokenizer_path, output_path, vocab_size):
    """
    Replace English tokenizer with Vietnamese-only tokenizer.

    This mode:
    - Loads pretrained model
    - Replaces tokenizer
    - Resizes embedding and output layers with random initialization
    - Preserves encoder weights (language-agnostic acoustic features)
    - Randomizes decoder embeddings (no English knowledge)

    Args:
        base_model_path: Path to pretrained .nemo model
        vietnamese_tokenizer_path: Path to Vietnamese tokenizer.model
        output_path: Output path for modified model
        vocab_size: Vietnamese vocabulary size
    """
    from nemo.collections.common.tokenizers import SentencePieceTokenizer

    logger.info("=" * 60)
    logger.info("Mode: Replace Tokenizer (Vietnamese-only)")
    logger.info("=" * 60)

    # Load base model
    logger.info(f"Loading base model from {base_model_path}")
    asr_model = ASRModel.restore_from(base_model_path, map_location='cpu')

    old_vocab_size = asr_model.decoder.prediction.embed.weight.shape[0]
    new_vocab_size = vocab_size + 1  # +1 for blank token
    embed_dim = 640

    logger.info(f"Original vocabulary: {old_vocab_size - 1} tokens")
    logger.info(f"Vietnamese vocabulary: {vocab_size} tokens")

    # Step 1: Replace tokenizer
    logger.info("Step 1: Replacing tokenizer with Vietnamese-only...")
    new_tokenizer = SentencePieceTokenizer(
        model_path=vietnamese_tokenizer_path)

    if new_tokenizer.vocab_size != vocab_size:
        logger.warning(
            f"Tokenizer vocab size mismatch: {new_tokenizer.vocab_size} != {vocab_size}")
        logger.warning("Using tokenizer's vocab size...")
        vocab_size = new_tokenizer.vocab_size
        new_vocab_size = vocab_size + 1

    asr_model.tokenizer = new_tokenizer
    logger.success(f"Tokenizer replaced: {vocab_size} Vietnamese tokens")

    # Step 2: Resize decoder embedding (random initialization)
    logger.info("Step 2: Resizing decoder embedding layer...")
    old_embed = asr_model.decoder.prediction.embed

    new_embed = nn.Embedding(
        num_embeddings=new_vocab_size,
        embedding_dim=embed_dim,
        padding_idx=new_vocab_size - 1  # Blank token at end
    )

    # Random initialization (no transfer from English)
    nn.init.xavier_uniform_(new_embed.weight)

    # Initialize blank token specially (small values)
    with torch.no_grad():
        new_embed.weight.data[new_vocab_size - 1] = torch.zeros(embed_dim)

    asr_model.decoder.prediction.embed = new_embed
    logger.success(f"Decoder embedding resized: {new_embed.weight.shape}")

    # Step 3: Resize joint output layer (random initialization)
    logger.info("Step 3: Resizing joint network output layer...")
    old_linear = asr_model.joint.joint_net[2]
    in_features = old_linear.in_features

    new_linear = nn.Linear(in_features, new_vocab_size)
    nn.init.xavier_uniform_(new_linear.weight)
    if new_linear.bias is not None:
        nn.init.zeros_(new_linear.bias)

    asr_model.joint.joint_net[2] = new_linear
    logger.success(f"Joint output resized: {new_linear.weight.shape}")

    # Step 4: Update model configuration
    logger.info("Step 4: Updating model configuration...")
    with open_dict(asr_model.cfg):
        asr_model.cfg.decoder.vocab_size = new_vocab_size
        if 'vocabulary' in asr_model.cfg.decoder:
            asr_model.cfg.decoder.pop('vocabulary')

        asr_model.cfg.joint.num_classes = new_vocab_size
        if 'vocabulary' in asr_model.cfg.joint:
            asr_model.cfg.joint.pop('vocabulary')

        if hasattr(asr_model.cfg, 'tokenizer'):
            asr_model.cfg.tokenizer.vocab_size = vocab_size
            if 'vocab_path' in asr_model.cfg.tokenizer:
                asr_model.cfg.tokenizer.pop('vocab_path')
            if 'spe_tokenizer_vocab' in asr_model.cfg.tokenizer:
                asr_model.cfg.tokenizer.pop('spe_tokenizer_vocab')

            import os
            new_path = os.path.abspath(vietnamese_tokenizer_path)
            asr_model.cfg.tokenizer.model_path = new_path
            try:
                asr_model.register_artifact("tokenizer.model_path", new_path)
            except Exception as e:
                logger.warning(f"Could not register artifact: {e}")

    logger.success("Model configuration updated")

    # Step 5: Save model
    logger.info(f"Step 5: Saving Vietnamese-only model to {output_path}")
    asr_model.save_to(output_path)

    logger.success("=" * 60)
    logger.success("Vietnamese-only model created successfully!")
    logger.success("=" * 60)
    logger.success(f"Vocabulary: {old_vocab_size - 1} → {vocab_size} tokens")
    logger.success(
        f"Decoder embedding: {old_embed.weight.shape} → {new_embed.weight.shape}")
    logger.success(
        f"Joint output: {old_linear.weight.shape} → {new_linear.weight.shape}")
    logger.success(
        f"Encoder: Preserved (acoustic features are language-agnostic)")
    logger.success(f"Output model: {output_path}")
    logger.info("")
    logger.info("Next step: Fine-tune on Vietnamese data with:")
    logger.info(f"  python scripts/finetune.py \\")
    logger.info(f"    --model_path {output_path} \\")
    logger.info(f"    --train_manifest data/train_mixed.json \\")
    logger.info(f"    --val_manifest data/val_mixed.json \\")
    logger.info(f"    --gpus 1 --max_steps 10000")


def full_scratch_mode(config_path, vietnamese_tokenizer_path, output_path):
    """
    Train from full scratch using NeMo config.

    This mode requires:
    - Custom NeMo YAML configuration
    - Much more training data (500+ hours recommended)
    - Longer training time (100K+ steps)

    Args:
        config_path: Path to NeMo model config YAML
        vietnamese_tokenizer_path: Path to Vietnamese tokenizer.model
        output_path: Output path for trained model
    """
    logger.error("Full scratch mode not yet implemented.")
    logger.info("For training from scratch with NeMo, please:")
    logger.info("1. Create a custom NeMo config YAML based on parakeet config")
    logger.info("2. Use NeMo's native training scripts")
    logger.info("3. See: https://github.com/NVIDIA/NeMo/tree/main/examples/asr")
    logger.info("")
    logger.info(
        "Recommended: Use 'replace' mode instead, which leverages pretrained weights.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Train Vietnamese-only multitalker ASR model from scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replace tokenizer (recommended - fast, leverages pretrained weights)
  python scripts/train_from_scratch.py \\
      --mode replace \\
      --base_model checkpoints/pretrained/multitalker-parakeet-streaming-0.6b-v1.nemo \\
      --vietnamese_tokenizer data/vi_tokenizer.model \\
      --vocab_size 2048 \\
      --output_model checkpoints/multitalker-vietnamese-only.nemo

  # Then fine-tune on Vietnamese data
  python scripts/finetune.py \\
      --model_path checkpoints/multitalker-vietnamese-only.nemo \\
      --train_manifest data/train_mixed.json \\
      --val_manifest data/val_mixed.json \\
      --gpus 1 --max_steps 10000
        """
    )

    parser.add_argument("--mode", type=str, required=True,
                        choices=['replace', 'full_scratch'],
                        help="Training mode: 'replace' (recommended) or 'full_scratch'")
    parser.add_argument("--base_model", type=str,
                        help="Path to base .nemo model (required for 'replace' mode)")
    parser.add_argument("--config", type=str,
                        help="Path to NeMo config YAML (required for 'full_scratch' mode)")
    parser.add_argument("--vietnamese_tokenizer", type=str, required=True,
                        help="Path to Vietnamese tokenizer.model")
    parser.add_argument("--vocab_size", type=int, default=2048,
                        help="Vietnamese vocabulary size (default: 2048)")
    parser.add_argument("--output_model", type=str, required=True,
                        help="Output path for model checkpoint")

    args = parser.parse_args()

    # Validate inputs based on mode
    if args.mode == 'replace':
        if not args.base_model:
            parser.error("--base_model is required for 'replace' mode")
        if not Path(args.base_model).exists():
            logger.error(f"Base model not found: {args.base_model}")
            sys.exit(1)
    elif args.mode == 'full_scratch':
        if not args.config:
            parser.error("--config is required for 'full_scratch' mode")
        if not Path(args.config).exists():
            logger.error(f"Config file not found: {args.config}")
            sys.exit(1)

    if not Path(args.vietnamese_tokenizer).exists():
        logger.error(
            f"Vietnamese tokenizer not found: {args.vietnamese_tokenizer}")
        sys.exit(1)

    # Execute based on mode
    if args.mode == 'replace':
        replace_tokenizer_mode(
            base_model_path=args.base_model,
            vietnamese_tokenizer_path=args.vietnamese_tokenizer,
            output_path=args.output_model,
            vocab_size=args.vocab_size
        )
    elif args.mode == 'full_scratch':
        full_scratch_mode(
            config_path=args.config,
            vietnamese_tokenizer_path=args.vietnamese_tokenizer,
            output_path=args.output_model
        )


if __name__ == "__main__":
    main()
