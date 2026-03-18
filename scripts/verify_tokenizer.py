#!/usr/bin/env python3
"""Verify extended tokenizer model."""

import argparse
import torch
from nemo.collections.asr.models import ASRModel
from loguru import logger

def verify_layer_dimensions(asr_model, expected_vocab_size):
    """Verify all layers have correct dimensions."""
    embed_size = asr_model.decoder.prediction.embed.weight.shape[0]
    joint_size = asr_model.joint.joint_net[2].out_features

    assert embed_size == expected_vocab_size, \
        f"Embedding size mismatch: {embed_size} != {expected_vocab_size}"
    assert joint_size == expected_vocab_size, \
        f"Joint output size mismatch: {joint_size} != {expected_vocab_size}"

    logger.success(f"✓ Layer dimensions correct (vocab_size={expected_vocab_size})")

def test_tokenization(asr_model):
    """Test tokenizer on mixed English/Vietnamese text."""
    test_samples = [
        "xin chào các bạn",  # Pure Vietnamese
        "tôi đang học tiếng việt",  # Pure Vietnamese
        "hello world",  # Pure English
        "tôi thích coffee",  # Vietlish (Vietnamese + English)
        "meeting lúc 3 giờ chiều",  # Vietlish
    ]

    tokenizer = asr_model.tokenizer

    for text in test_samples:
        tokens = tokenizer.text_to_ids(text)
        reconstructed = tokenizer.ids_to_text(tokens)

        unk_count = sum(1 for t in tokens if t == tokenizer.unk_id)
        unk_rate = unk_count / len(tokens) if tokens else 0

        logger.info(f"Text: {text}")
        logger.info(f"  Tokens ({len(tokens)}): {tokens[:10]}...")
        logger.info(f"  UNK rate: {unk_rate:.1%}")
        logger.info(f"  Reconstructed: {reconstructed}")
        logger.info("")

    logger.success("✓ Tokenization test completed")

def test_forward_pass(asr_model):
    """Test forward pass with dummy audio."""
    batch_size = 1
    mel_features = 128
    seq_len = 100

    dummy_input = torch.randn(batch_size, mel_features, seq_len)
    dummy_lengths = torch.tensor([seq_len])

    with torch.no_grad():
        encoded, encoded_len = asr_model.encoder(
            audio_signal=dummy_input,
            length=dummy_lengths
        )

    logger.success(f"✓ Forward pass successful. Encoded shape: {encoded.shape}")

def main():
    parser = argparse.ArgumentParser(description="Verify extended tokenizer model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to extended .nemo model")
    parser.add_argument("--vocab_size", type=int, required=True,
                       help="Expected vocabulary size (including blank)")

    args = parser.parse_args()

    logger.info(f"Loading model from {args.model_path}")
    asr_model = ASRModel.restore_from(args.model_path, map_location='cpu')

    logger.info("=" * 60)
    logger.info("Test 1: Verifying layer dimensions...")
    verify_layer_dimensions(asr_model, args.vocab_size)

    logger.info("=" * 60)
    logger.info("Test 2: Testing tokenization on mixed text...")
    test_tokenization(asr_model)

    logger.info("=" * 60)
    logger.info("Test 3: Testing forward pass...")
    test_forward_pass(asr_model)

    logger.info("=" * 60)
    logger.success("All verification tests passed! ✓")
    logger.success("Model is ready for fine-tuning.")

if __name__ == "__main__":
    main()
