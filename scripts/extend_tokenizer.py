#!/usr/bin/env python3
"""
Extend multitalker ASR model with Vietnamese tokenizer.

This script:
1. Loads the pretrained English ASR model
2. Loads merged vocabulary and token mapping
3. Resizes decoder embedding and joint output layers
4. Updates model configuration
5. Saves extended model as new .nemo checkpoint

Usage:
    python scripts/extend_tokenizer.py \
        --base_model checkpoints/pretrained/multitalker-parakeet-streaming-0.6b-v1.nemo \
        --merged_vocab data/merged_vocab.txt \
        --token_mapping data/token_mapping.json \
        --merged_tokenizer data/merged_tokenizer.model \
        --output_model checkpoints/multitalker-vietnamese-extended-0.6b-v1.nemo
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nemo.collections.asr.models import ASRModel
from multitalker_asr.tokenizer_utils import TokenizerExtender
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="Extend ASR model with Vietnamese tokenizer")
    parser.add_argument("--base_model", type=str, required=True,
                       help="Path to base .nemo model")
    parser.add_argument("--merged_vocab", type=str, required=True,
                       help="Path to merged vocabulary file")
    parser.add_argument("--token_mapping", type=str, required=True,
                       help="Path to token mapping JSON")
    parser.add_argument("--merged_tokenizer", type=str, required=True,
                       help="Path to retrained merged tokenizer.model")
    parser.add_argument("--output_model", type=str, required=True,
                       help="Output path for extended .nemo model")

    args = parser.parse_args()

    # Validate inputs
    for path_arg in ['base_model', 'merged_vocab', 'token_mapping', 'merged_tokenizer']:
        if not Path(getattr(args, path_arg)).exists():
            logger.error(f"{path_arg} not found: {getattr(args, path_arg)}")
            sys.exit(1)

    # Load base model
    logger.info(f"Loading base model from {args.base_model}")
    asr_model = ASRModel.restore_from(args.base_model, map_location='cpu')

    old_embed_size = asr_model.decoder.prediction.embed.weight.shape[0]
    old_joint_size = asr_model.joint.joint_net[2].out_features
    logger.info(f"Original embedding size: {old_embed_size}")
    logger.info(f"Original joint output size: {old_joint_size}")

    # Initialize extender
    logger.info("Initializing tokenizer extender...")
    extender = TokenizerExtender(
        asr_model=asr_model,
        merged_vocab_path=args.merged_vocab,
        token_mapping_path=args.token_mapping
    )

    # Extend model
    logger.info("Step 1: Resizing decoder embedding...")
    extender.resize_decoder_embedding()

    logger.info("Step 2: Resizing joint output layer...")
    extender.resize_joint_output()

    logger.info("Step 3: Rebuilding tokenizer...")
    extender.rebuild_tokenizer(args.merged_tokenizer)

    logger.info("Step 4: Updating model configuration...")
    extender.update_tokenizer_config()

    # Save extended model
    logger.info(f"Step 5: Saving extended model to {args.output_model}")
    extender.save_extended_model(args.output_model)

    # Summary
    new_embed_size = asr_model.decoder.prediction.embed.weight.shape[0]
    new_joint_size = asr_model.joint.joint_net[2].out_features

    logger.success("=" * 60)
    logger.success("Tokenizer extension completed successfully!")
    logger.success("=" * 60)
    logger.success(f"Decoder embedding: {old_embed_size} → {new_embed_size}")
    logger.success(f"Joint output: {old_joint_size} → {new_joint_size}")
    logger.success(f"Vocabulary: {extender.old_vocab_size} → {extender.new_vocab_size}")
    logger.success(f"Extended model: {args.output_model}")

if __name__ == "__main__":
    main()
