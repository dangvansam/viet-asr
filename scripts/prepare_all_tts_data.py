#!/usr/bin/env python3
"""
Prepare all TTS datasets for multitalker ASR training.
Converts pipe-separated metadata files to NeMo JSON manifests.
"""

import os
import json
import argparse
import librosa
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import random


def parse_metadata_line(line: str, dataset_dir: str) -> dict | None:
    """Parse a pipe-separated metadata line.

    Format: speaker_id|audio_path|text
    """
    parts = line.strip().split('|')
    if len(parts) < 3:
        return None

    speaker_id = parts[0]
    original_audio_path = parts[1]
    text = parts[2]

    # Extract filename and relative path from original path
    audio_filename = os.path.basename(original_audio_path)

    # Get relative path from wavs directory (for nested structures like multi_speakers)
    if "/wavs/" in original_audio_path:
        relative_path = original_audio_path.split("/wavs/")[-1]
    else:
        relative_path = audio_filename

    # Try multiple possible locations for audio files
    possible_paths = [
        os.path.join(dataset_dir, "wavs", relative_path),  # Nested structure
        os.path.join(dataset_dir, "wavs", audio_filename),  # Flat structure
        os.path.join(dataset_dir, audio_filename),
        os.path.join(dataset_dir, "audio", audio_filename),
    ]

    audio_path = None
    for p in possible_paths:
        if os.path.exists(p):
            audio_path = os.path.abspath(p)
            break

    if audio_path is None:
        return None

    return {
        "speaker_id": speaker_id,
        "audio_path": audio_path,
        "text": text
    }


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    try:
        duration = librosa.get_duration(path=audio_path)
        return round(duration, 4)
    except Exception as e:
        logger.warning(f"Could not get duration for {audio_path}: {e}")
        return 0.0


def process_dataset(dataset_dir: str, metadata_file: str = "metadata.txt") -> list:
    """Process a single dataset directory."""
    metadata_path = os.path.join(dataset_dir, metadata_file)

    if not os.path.exists(metadata_path):
        logger.warning(f"Metadata file not found: {metadata_path}")
        return []

    entries = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    dataset_name = os.path.basename(dataset_dir)
    logger.info(f"Processing {dataset_name}: {len(lines)} lines")

    for line in tqdm(lines, desc=dataset_name):
        parsed = parse_metadata_line(line, dataset_dir)
        if parsed:
            duration = get_audio_duration(parsed["audio_path"])
            if duration > 0.5:  # Filter out very short audio
                entry = {
                    "audio_filepath": parsed["audio_path"],
                    "offset": 0.0,
                    "duration": duration,
                    "label": parsed["speaker_id"],
                    "text": parsed["text"],
                    "num_speakers": 1
                }
                entries.append(entry)

    logger.info(f"Processed {len(entries)} valid entries from {dataset_name}")
    return entries


def save_manifest(entries: list, output_path: str):
    """Save entries as NeMo JSON manifest (JSONL format)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(entries)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare all TTS datasets for training")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing TTS datasets")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory for manifests")
    parser.add_argument("--val_split", type=float, default=0.05,
                        help="Validation split ratio (default: 0.05)")
    parser.add_argument("--skip_duration", action="store_true",
                        help="Skip duration calculation (faster but less accurate)")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Specific datasets to process (default: all)")

    args = parser.parse_args()

    # Find all dataset directories
    data_root = Path(args.data_root)

    if args.datasets:
        dataset_dirs = [data_root / d for d in args.datasets]
    else:
        # Auto-discover datasets with metadata.txt
        dataset_dirs = [
            d for d in data_root.iterdir()
            if d.is_dir() and (d / "metadata.txt").exists()
        ]

    logger.info(f"Found {len(dataset_dirs)} datasets to process")

    # Process all datasets
    all_entries = []
    for dataset_dir in dataset_dirs:
        entries = process_dataset(str(dataset_dir))
        all_entries.extend(entries)

    logger.info(f"Total entries collected: {len(all_entries)}")

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_entries)

    val_size = int(len(all_entries) * args.val_split)
    val_entries = all_entries[:val_size]
    train_entries = all_entries[val_size:]

    # Save manifests
    output_dir = Path(args.output_dir)
    save_manifest(train_entries, str(output_dir / "train_single_speaker.json"))
    save_manifest(val_entries, str(output_dir / "val_single_speaker.json"))

    # Print statistics
    speakers = set(e["label"] for e in all_entries)
    total_duration = sum(e["duration"] for e in all_entries)

    logger.info("=" * 50)
    logger.info("Dataset Statistics:")
    logger.info(f"  Total samples: {len(all_entries)}")
    logger.info(f"  Training samples: {len(train_entries)}")
    logger.info(f"  Validation samples: {len(val_entries)}")
    logger.info(f"  Unique speakers: {len(speakers)}")
    logger.info(f"  Total duration: {total_duration / 3600:.2f} hours")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
