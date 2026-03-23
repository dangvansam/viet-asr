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
    old_embed_size = old_embed.weight.shape[0]

    new_embed = nn.Embedding(
        num_embeddings=new_vocab_size,
        embedding_dim=embed_dim,
        padding_idx=new_vocab_size - 1  # Blank token at end
    )

    # Random initialization (no transfer from English)
    nn.init.xavier_uniform_(new_embed.weight)

    # CRITICAL FIX: Transfer pretrained blank token embedding to prevent RNN-T posterior collapse
    with torch.no_grad():
        new_embed.weight.data[new_vocab_size -
                              1] = old_embed.weight.data[old_embed_size - 1].clone()

    asr_model.decoder.prediction.embed = new_embed
    logger.success(f"Decoder embedding resized: {new_embed.weight.shape}")

    # Step 3: Resize joint output layer (random initialization)
    logger.info("Step 3: Resizing joint network output layer...")
    old_linear = asr_model.joint.joint_net[2]
    old_out_features = old_linear.out_features
    in_features = old_linear.in_features

    new_linear = nn.Linear(in_features, new_vocab_size)

    with torch.no_grad():
        # Initialize Vietnamese token weights with small variance
        nn.init.normal_(new_linear.weight, mean=0.0, std=0.01)

        # CRITICAL FIX: Transfer pretrained blank token weights
        new_linear.weight.data[new_vocab_size -
                               1] = old_linear.weight.data[old_out_features - 1].clone()

        if new_linear.bias is not None and old_linear.bias is not None:
            # Initialize token biases to very negative to favor pretrained blank early in training
            nn.init.constant_(new_linear.bias, -20.0)

            # CRITICAL FIX: Transfer pretrained blank token bias
            new_linear.bias.data[new_vocab_size -
                                 1] = old_linear.bias.data[old_out_features - 1].clone()

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


def full_scratch_mode(config_path, vietnamese_tokenizer_path, output_path, vocab_size):
    """
    Train from full scratch using NeMo config.

    This mode:
    - Loads custom NeMo YAML configuration
    - Injects the chosen Vietnamese Tokenizer directly into the config
    - Initializes the model with random weights mapped to the new token size
    """
    from nemo.collections.common.tokenizers import SentencePieceTokenizer
    import os

    logger.info("=" * 60)
    logger.info("Mode: Full Scratch (Vietnamese-only)")
    logger.info("=" * 60)

    # 1. Load config
    logger.info(f"Loading config from {config_path}")
    from omegaconf import OmegaConf, open_dict
    cfg = OmegaConf.load(config_path)

    # 2. Check tokenizer
    logger.info(
        f"Loading Vietnamese tokenizer from {vietnamese_tokenizer_path}")
    tokenizer = SentencePieceTokenizer(model_path=vietnamese_tokenizer_path)

    if tokenizer.vocab_size != vocab_size:
        logger.warning(
            f"Tokenizer vocab size mismatch: {tokenizer.vocab_size} != {vocab_size}")
        vocab_size = tokenizer.vocab_size

    new_vocab_size = vocab_size + 1

    # 3. Patch config
    logger.info("Patching configuration for fresh model...")
    with open_dict(cfg):
        if 'tokenizer' not in cfg:
            cfg.tokenizer = {}

        new_path = os.path.abspath(vietnamese_tokenizer_path)
        cfg.tokenizer.dir = os.path.dirname(new_path)
        cfg.tokenizer.type = "bpe"
        cfg.tokenizer.model_path = new_path
        cfg.tokenizer.vocab_size = vocab_size

        if 'spe_tokenizer_vocab' in cfg.tokenizer:
            cfg.tokenizer.pop('spe_tokenizer_vocab')

        # Point to the actual vocab file
        vocab_file = new_path.replace(".model", ".vocab")
        if os.path.exists(vocab_file):
            cfg.tokenizer.vocab_path = vocab_file
        else:
            if 'vocab_path' in cfg.tokenizer:
                cfg.tokenizer.pop('vocab_path')

        if 'decoder' in cfg:
            cfg.decoder.vocab_size = new_vocab_size
            if 'vocabulary' in cfg.decoder:
                cfg.decoder.pop('vocabulary')

        if 'joint' in cfg:
            cfg.joint.num_classes = new_vocab_size
            if 'vocabulary' in cfg.joint:
                cfg.joint.pop('vocabulary')

        # Ensure model defaults also match if present
        if 'model_defaults' in cfg:
            if 'vocab_size' in cfg.model_defaults:
                cfg.model_defaults.vocab_size = vocab_size
            if 'num_classes' in cfg.model_defaults:
                cfg.model_defaults.num_classes = new_vocab_size

        # Remove datasets to prevent dataloader initialization failure
        ds_configs = {}
        for ds in ['train_ds', 'validation_ds', 'test_ds']:
            if ds in cfg:
                ds_configs[ds] = cfg.pop(ds)

    # 4. Initialize model
    logger.info("Initializing ASRModel from config (this may take a moment)...")
    from nemo.collections.asr.models import EncDecMultiTalkerRNNTBPEModel

    try:
        asr_model = EncDecMultiTalkerRNNTBPEModel(cfg=cfg, trainer=None)
    except Exception as e:
        logger.error(f"Failed to initialize model from config: {e}")
        import sys
        sys.exit(1)

    # Sync config with actual dimensions after initialization
    with open_dict(asr_model.cfg):
        if 'decoder' in asr_model.cfg:
            asr_model.cfg.decoder.vocab_size = new_vocab_size
        if 'joint' in asr_model.cfg:
            asr_model.cfg.joint.num_classes = new_vocab_size
        if 'model_defaults' in asr_model.cfg:
            asr_model.cfg.model_defaults.num_classes = new_vocab_size

    # Restore datasets in config before saving
    with open_dict(asr_model.cfg):
        for ds, ds_cfg in ds_configs.items():
            asr_model.cfg[ds] = ds_cfg

    # Register tokenizer artifact
    try:
        asr_model.register_artifact(
            "tokenizer.model_path", cfg.tokenizer.model_path)
    except Exception as e:
        logger.warning(f"Could not register artifact: {e}")

    # 5. Save model
    logger.info(f"Saving initialized model to {output_path}")
    asr_model.save_to(output_path)

    logger.success("=" * 60)
    logger.success("Scratch model created successfully!")
    logger.success(f"Vocabulary: {vocab_size} tokens")
    logger.success(f"Output model: {output_path}")
    logger.info("Next step: Train on Vietnamese data using finetune.py")


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
            output_path=args.output_model,
            vocab_size=args.vocab_size
        )


if __name__ == "__main__":
    main()
