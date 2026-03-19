#!/usr/bin/env python3
"""Extract Vietnamese text from training manifests for tokenizer training."""

import argparse
import json
from pathlib import Path
from loguru import logger

def extract_text_from_manifest(manifest_path, output_path):
    """Extract all text fields from manifest to a corpus file."""
    texts = []

    with open(manifest_path, 'r', encoding='utf-8') as f:
        try:
            # Try loading as JSON array first (Lhotse CutSet format)
            content = f.read()
            data = json.loads(content)

            # Handle JSON array
            if isinstance(data, list):
                for item in data:
                    if 'supervisions' in item:
                        for sup in item['supervisions']:
                            if 'text' in sup:
                                texts.append(sup['text'])
                    elif 'text' in item:
                        texts.append(item['text'])
            # Handle single JSON object
            elif isinstance(data, dict):
                if 'text' in data:
                    texts.append(data['text'])
                elif 'supervisions' in data:
                    for sup in data['supervisions']:
                        if 'text' in sup:
                            texts.append(sup['text'])
        except json.JSONDecodeError:
            # Fall back to JSONL format (one JSON per line)
            f.seek(0)
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'text' in data:
                        texts.append(data['text'])
                    elif 'supervisions' in data:
                        for sup in data['supervisions']:
                            if 'text' in sup:
                                texts.append(sup['text'])
                except json.JSONDecodeError:
                    continue

    # Write corpus (one sentence per line)
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text.strip() + '\n')

    logger.info(f"Extracted {len(texts)} utterances to {output_path}")
    return len(texts)

def main():
    parser = argparse.ArgumentParser(description="Prepare Vietnamese corpus for tokenizer training")
    parser.add_argument("--manifests", type=str, nargs='+', required=True,
                       help="Path(s) to training manifest files")
    parser.add_argument("--output", type=str, required=True,
                       help="Output corpus file path")

    args = parser.parse_args()

    all_texts = []
    for manifest_path in args.manifests:
        logger.info(f"Processing {manifest_path}...")
        extract_text_from_manifest(manifest_path, f"{args.output}.tmp")
        with open(f"{args.output}.tmp", 'r', encoding='utf-8') as f:
            all_texts.extend(f.readlines())

    # Combine and deduplicate
    unique_texts = list(set(all_texts))
    with open(args.output, 'w', encoding='utf-8') as f:
        f.writelines(unique_texts)

    logger.success(f"Total unique utterances: {len(unique_texts)}")

if __name__ == "__main__":
    main()
