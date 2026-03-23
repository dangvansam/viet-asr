import argparse
import os
import sentencepiece as spm
from lhotse import CutSet
from loguru import logger


def extract_text_from_manifest(manifest_path, output_text_file):
    logger.info(f"Extracting text from {manifest_path}...")
    cuts = CutSet.from_file(manifest_path)
    count = 0
    with open(output_text_file, "w", encoding="utf-8") as f:
        for cut in cuts:
            for sup in cut.supervisions:
                if sup.text:
                    f.write(sup.text.strip() + "\n")
                    count += 1
    logger.success(f"Extracted {count} text lines to {output_text_file}")
    return count


def train_tokenizer(text_file, output_dir, model_prefix=None, vocab_size=1024, model_type="bpe"):
    os.makedirs(output_dir, exist_ok=True)
    if not model_prefix:
        model_prefix = os.path.join(output_dir, "tokenizer")
    logger.info(
        f"Training SentencePiece {model_type} tokenizer with vocab_size={vocab_size}...")

    # Train SPM
    spm.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        # Important for Vietnamese to include all characters, no bytes fallback needed if 1.0 covers it, but BPE covers everything
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
        # Required for NeMo compatibility
        byte_fallback=False,
    )

    logger.success(f"Tokenizer trained and saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece tokenizer on a dataset manifest")
    parser.add_argument("--manifest", type=str,
                        help="Path to NeMo or Lhotse train manifest (JSON or JSONL)")
    parser.add_argument("--input", type=str,
                        help="Raw text file to train on (skips manifest extraction)")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Directory to save the trained tokenizer. Default is data/")
    parser.add_argument("--model_prefix", type=str,
                        help="Specific prefix for model output (overrides output_dir)")
    parser.add_argument("--vocab_size", type=int, default=1024,
                        help="Vocabulary size for the tokenizer")
    parser.add_argument("--model_type", type=str, default="bpe",
                        choices=["bpe", "unigram"], help="SentencePiece model type")

    args = parser.parse_args()
    
    if not args.manifest and not args.input:
        parser.error("Either --manifest or --input must be provided.")

    if args.input:
        temp_text = args.input
        logger.info(f"Using provided text file: {args.input}")
    else:
        # Temp file to hold extracted text
        temp_text = os.path.join(args.output_dir, "train_text.txt")
        os.makedirs(args.output_dir, exist_ok=True)

        # Simple check if it's Lhotse or NeMo, Lhotse cuts might be used directly if it's Lhotse
        try:
            extract_text_from_manifest(args.manifest, temp_text)
        except Exception as e:
            logger.warning(
                f"Failed to use Lhotse CutSet (Error: {e}), falling back to direct JSON reading...")
            import json
            with open(args.manifest, "r", encoding="utf-8") as fin, open(temp_text, "w", encoding="utf-8") as fout:
                count = 0
                for line in fin:
                    data = json.loads(line.strip())
                    if 'text' in data:
                        fout.write(data['text'].strip() + "\n")
                        count += 1
            logger.info(f"Extracted {count} lines from NeMo manifest.")

    if os.path.exists(temp_text) and os.path.getsize(temp_text) > 0:
        train_tokenizer(temp_text, args.output_dir,
                        args.model_prefix, args.vocab_size, args.model_type)
        logger.info("Done!")
    else:
        logger.error(f"No text extracted or file {temp_text} is empty.")
