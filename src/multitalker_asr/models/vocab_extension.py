import os
import shutil
from typing import List

from loguru import logger


class VocabularyExtender:
    """Extends RNNT vocabulary with special tokens (<EOU>, punctuation, etc.)."""

    SPECIAL_TOKENS = {
        "<EOU>": "end_of_utterance",
    }

    @staticmethod
    def create_extended_tokenizer(
        base_tokenizer_dir: str,
        output_dir: str,
        special_tokens: List[str],
    ) -> str:
        """Create a new tokenizer directory with added special tokens.

        For SentencePiece tokenizers, creates a user_defined_symbols file
        that NeMo's change_vocabulary() can consume.

        Args:
            base_tokenizer_dir: Path to existing tokenizer directory
            output_dir: Where to write the extended tokenizer
            special_tokens: List of tokens to add (e.g., ["<EOU>"])

        Returns:
            output_dir path
        """
        os.makedirs(output_dir, exist_ok=True)

        # Copy base tokenizer files
        if os.path.isfile(base_tokenizer_dir):
            base_dir = os.path.dirname(base_tokenizer_dir)
        else:
            base_dir = base_tokenizer_dir

        for f in os.listdir(base_dir):
            src = os.path.join(base_dir, f)
            dst = os.path.join(output_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

        # Read existing vocab to check for duplicates
        vocab_path = None
        for name in ["tokenizer.vocab", "vocab.txt"]:
            candidate = os.path.join(output_dir, name)
            if os.path.exists(candidate):
                vocab_path = candidate
                break

        existing_tokens = set()
        if vocab_path:
            with open(vocab_path, "r", encoding="utf-8") as f:
                for line in f:
                    token = line.strip().split("\t")[0]
                    existing_tokens.add(token)

        # Add special tokens
        added = []
        for token in special_tokens:
            if token in existing_tokens:
                logger.info(f"Token '{token}' already exists in vocab, skipping")
                continue
            added.append(token)

        if added and vocab_path:
            with open(vocab_path, "a", encoding="utf-8") as f:
                for token in added:
                    f.write(f"{token}\t0\n")
            logger.info(f"Added {len(added)} tokens to {vocab_path}: {added}")

        # Write user_defined_symbols file (NeMo convention)
        uds_path = os.path.join(output_dir, "user_defined_symbols.txt")
        with open(uds_path, "w", encoding="utf-8") as f:
            for token in added:
                f.write(f"{token}\n")

        logger.success(f"Extended tokenizer created at {output_dir}")
        return output_dir

    @staticmethod
    def extend_rnnt_vocab(
        model,
        tokenizer_dir: str,
    ) -> int:
        """Extend RNNT model vocabulary using NeMo's change_vocabulary.

        Args:
            model: NeMo EncDecRNNTBPEModel or MultitalkerASRModel
            tokenizer_dir: Path to the extended tokenizer directory

        Returns:
            New vocab size
        """
        if hasattr(model, "change_vocabulary"):
            model.change_vocabulary(tokenizer_dir)
        elif hasattr(model, "asr_model"):
            model.asr_model.change_vocabulary(
                new_tokenizer_dir=tokenizer_dir,
                new_tokenizer_type="bpe",
            )
        else:
            raise RuntimeError("Model does not support change_vocabulary()")

        new_size = model.tokenizer.vocab_size if hasattr(model, "tokenizer") else -1
        logger.success(f"Vocabulary extended, new size: {new_size}")
        return new_size
