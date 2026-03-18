#!/usr/bin/env python3
"""Train Vietnamese SentencePiece tokenizer."""

import argparse
import sentencepiece as spm
from loguru import logger

def train_vietnamese_tokenizer(corpus_path, output_prefix, vocab_size):
    """Train SentencePiece model optimized for Vietnamese."""

    # Training arguments optimized for Vietnamese
    training_args = f"""
        --input={corpus_path}
        --model_prefix={output_prefix}
        --vocab_size={vocab_size}
        --character_coverage=1.0
        --model_type=bpe
        --pad_id=0
        --unk_id=1
        --bos_id=-1
        --eos_id=-1
        --user_defined_symbols=
        --normalization_rule_name=identity
        --remove_extra_whitespaces=false
        --split_by_unicode_script=true
        --split_by_whitespace=true
        --byte_fallback=true
    """.replace('\n', ' ').strip()

    logger.info(f"Training Vietnamese tokenizer with vocab_size={vocab_size}")
    logger.info(f"Input corpus: {corpus_path}")

    spm.SentencePieceTrainer.train(training_args)

    logger.success(f"Tokenizer saved to {output_prefix}.model and {output_prefix}.vocab")

def main():
    parser = argparse.ArgumentParser(description="Train Vietnamese SentencePiece tokenizer")
    parser.add_argument("--input", type=str, required=True,
                       help="Input corpus file")
    parser.add_argument("--model_prefix", type=str, required=True,
                       help="Output model prefix (e.g., 'data/vi_tokenizer')")
    parser.add_argument("--vocab_size", type=int, default=1024,
                       help="Vocabulary size for Vietnamese tokenizer (default: 1024)")

    args = parser.parse_args()

    train_vietnamese_tokenizer(args.input, args.model_prefix, args.vocab_size)

if __name__ == "__main__":
    main()
