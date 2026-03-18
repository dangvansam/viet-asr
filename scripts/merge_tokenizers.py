#!/usr/bin/env python3
"""Merge English and Vietnamese SentencePiece vocabularies."""

import argparse
import json
import sentencepiece as spm
from loguru import logger
from pathlib import Path

def load_vocab_from_model(model_path):
    """Load vocabulary from SentencePiece model."""
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    vocab = []
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        vocab.append((piece, score))

    return vocab

def extract_vocab_from_nemo(nemo_path, temp_dir="temp_tokenizer"):
    """Extract tokenizer from .nemo checkpoint."""
    import tarfile
    import os

    Path(temp_dir).mkdir(exist_ok=True)

    # Extract .nemo (tar archive)
    with tarfile.open(nemo_path, 'r') as tar:
        # Find tokenizer.model file
        for member in tar.getmembers():
            if 'tokenizer.model' in member.name:
                tar.extract(member, temp_dir)
                tokenizer_path = os.path.join(temp_dir, member.name)
                logger.info(f"Extracted tokenizer to {tokenizer_path}")
                return tokenizer_path

    raise FileNotFoundError("No tokenizer.model found in .nemo checkpoint")

def merge_vocabularies(english_vocab, vietnamese_vocab, output_path):
    """Merge two vocabularies, avoiding duplicates."""

    # Keep all English tokens
    merged_vocab = list(english_vocab)
    english_pieces = set(piece for piece, _ in english_vocab)

    # Add Vietnamese tokens that don't exist in English vocab
    added_count = 0
    for piece, score in vietnamese_vocab:
        # Skip special tokens (<unk>, <s>, </s>, etc.) from Vietnamese
        if piece.startswith('<') and piece.endswith('>'):
            continue

        if piece not in english_pieces:
            merged_vocab.append((piece, score))
            added_count += 1

    logger.info(f"English tokens: {len(english_vocab)}")
    logger.info(f"Vietnamese tokens added: {added_count}")
    logger.info(f"Total merged tokens: {len(merged_vocab)}")

    # Save merged vocabulary
    with open(output_path, 'w', encoding='utf-8') as f:
        for piece, score in merged_vocab:
            f.write(f"{piece}\t{score}\n")

    return len(merged_vocab)

def main():
    parser = argparse.ArgumentParser(description="Merge English and Vietnamese tokenizers")
    parser.add_argument("--english_model", type=str, required=True,
                       help="Path to .nemo checkpoint or English tokenizer.model")
    parser.add_argument("--vietnamese_model", type=str, required=True,
                       help="Path to Vietnamese tokenizer.model")
    parser.add_argument("--output_vocab", type=str, required=True,
                       help="Output merged vocabulary file")
    parser.add_argument("--output_mapping", type=str, required=True,
                       help="Output JSON mapping file (old_id -> new_id)")

    args = parser.parse_args()

    # Load English vocabulary
    if args.english_model.endswith('.nemo'):
        logger.info("Extracting tokenizer from .nemo checkpoint...")
        english_tokenizer_path = extract_vocab_from_nemo(args.english_model)
    else:
        english_tokenizer_path = args.english_model

    logger.info("Loading English vocabulary...")
    english_vocab = load_vocab_from_model(english_tokenizer_path)

    # Load Vietnamese vocabulary
    logger.info("Loading Vietnamese vocabulary...")
    vietnamese_vocab = load_vocab_from_model(args.vietnamese_model)

    # Merge vocabularies
    logger.info("Merging vocabularies...")
    total_vocab_size = merge_vocabularies(english_vocab, vietnamese_vocab, args.output_vocab)

    # Create token ID mapping (for transferring embeddings)
    # English tokens keep their original IDs
    token_mapping = {i: i for i in range(len(english_vocab))}

    with open(args.output_mapping, 'w') as f:
        json.dump({
            'old_vocab_size': len(english_vocab),
            'new_vocab_size': total_vocab_size,
            'token_mapping': token_mapping,
            'vietnamese_tokens_start': len(english_vocab)
        }, f, indent=2)

    logger.success(f"Merged vocabulary saved to {args.output_vocab}")
    logger.success(f"Token mapping saved to {args.output_mapping}")

if __name__ == "__main__":
    main()
