import argparse
import os

import sentencepiece as spm
from loguru import logger

from multitalker_asr.utils import TextExtractor


class TokenizerTrainer:
    def __init__(
        self,
        output_dir: str,
        vocab_size: int = 1024,
        model_type: str = "bpe",
        model_prefix: str = None,
    ):
        self._output_dir = output_dir
        self._vocab_size = vocab_size
        self._model_type = model_type
        self._model_prefix = model_prefix or os.path.join(output_dir, "tokenizer")

    def train(self, text_file: str) -> None:
        os.makedirs(self._output_dir, exist_ok=True)
        logger.info(f"Training SentencePiece {self._model_type} tokenizer with vocab_size={self._vocab_size}...")

        spm.SentencePieceTrainer.train(
            input=text_file,
            model_prefix=self._model_prefix,
            vocab_size=self._vocab_size,
            model_type=self._model_type,
            character_coverage=1.0,
            pad_id=-1,
            unk_id=0,
            bos_id=-1,
            eos_id=-1,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
            user_defined_symbols=[],
            byte_fallback=False,
        )

        logger.success(f"Tokenizer trained and saved to {self._output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer")
    parser.add_argument("--manifest", type=str)
    parser.add_argument("--input", type=str)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--model_prefix", type=str)
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--model_type", type=str, default="bpe", choices=["bpe", "unigram"])

    args = parser.parse_args()

    if not args.manifest and not args.input:
        parser.error("Either --manifest or --input must be provided.")

    if args.input:
        temp_text = args.input
        logger.info(f"Using provided text file: {args.input}")
    else:
        temp_text = os.path.join(args.output_dir, "train_text.txt")
        os.makedirs(args.output_dir, exist_ok=True)

        extractor = TextExtractor()
        extractor.extract(args.manifest, output_path=temp_text)

    if os.path.exists(temp_text) and os.path.getsize(temp_text) > 0:
        trainer = TokenizerTrainer(
            output_dir=args.output_dir,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            model_prefix=args.model_prefix,
        )
        trainer.train(temp_text)
        logger.info("Done!")
    else:
        logger.error(f"No text extracted or file {temp_text} is empty.")
