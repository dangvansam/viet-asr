#!/usr/bin/env python3
"""
Prepare multi-task training manifests with paralinguistic labels.

Extends existing single-speaker manifests with:
  - gender: extracted from speaker_id/directory name (nu-=female, nam-=male)
  - age: default "young" (TTS voices are typically young adults)
  - emotion: default "neutral" (TTS data is neutral reading)
  - voice_state: default "sober"
  - language: "vi" for all Vietnamese TTS data

Supports:
  1. Auto-annotation from existing manifests (extract from speaker_id patterns)
  2. Custom speaker metadata mapping file (JSON)
  3. LLM-based pseudo-labeling for emotion (future)

Usage:
    # Auto-annotate from speaker_id patterns
    uv run scripts/prepare_multitask_data.py \
        --input_manifest data/train_single_speaker.json \
        --output_manifest data/train_multitask.json

    # With custom speaker metadata
    uv run scripts/prepare_multitask_data.py \
        --input_manifest data/train_single_speaker.json \
        --output_manifest data/train_multitask.json \
        --speaker_metadata data/speaker_metadata.json
"""

import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, Optional

from loguru import logger


# Auto-detection rules for gender from speaker_id / directory name
GENDER_PATTERNS = {
    "female": [
        r"^nu[-_]",           # nu-mien-bac, nu_mien_nam
        r"female",
        r"Linh_Elena",
        r"Minh_Phuong",
        r"MinhHoa",
    ],
    "male": [
        r"^nam[-_]",          # nam-mien-bac, nam_mien_nam
        r"male",
        r"quockhanh",
        r"hoanganhquan",
        r"quan_doi",
    ],
}

# Known speaker metadata (can be extended)
KNOWN_SPEAKERS: Dict[str, Dict[str, str]] = {
    "nu-mien-bac": {"gender": "female", "age": "young", "language": "vi"},
    "nu-mien-nam": {"gender": "female", "age": "young", "language": "vi"},
    "nam-mien-bac": {"gender": "male", "age": "young", "language": "vi"},
    "nam-mien-nam": {"gender": "male", "age": "young", "language": "vi"},
    "nam-quan-doi": {"gender": "male", "age": "middle_age", "language": "vi"},
}

# Default labels for unknown speakers
DEFAULTS = {
    "emotion": "neutral",
    "gender": "male",
    "age": "young",
    "voice_state": "sober",
    "language": "vi",
}


def detect_gender(speaker_id: str, audio_path: str = "") -> str:
    """Auto-detect gender from speaker_id or audio path."""
    search_str = f"{speaker_id} {audio_path}".lower()

    for gender, patterns in GENDER_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, search_str, re.IGNORECASE):
                return gender

    return DEFAULTS["gender"]


def detect_age(speaker_id: str) -> str:
    """Auto-detect age group from speaker_id."""
    sid = speaker_id.lower()
    if "child" in sid or "tre_em" in sid:
        return "child"
    if "old" in sid or "gia" in sid:
        return "old"
    if "quan_doi" in sid or "quan-doi" in sid:
        return "middle_age"
    return "young"


def get_speaker_metadata(
    speaker_id: str,
    audio_path: str,
    custom_metadata: Optional[Dict] = None,
) -> Dict[str, str]:
    """Get full metadata for a speaker, using multiple sources."""
    # Priority 1: Custom metadata file
    if custom_metadata and speaker_id in custom_metadata:
        meta = {**DEFAULTS, **custom_metadata[speaker_id]}
        return meta

    # Priority 2: Known speakers table
    if speaker_id in KNOWN_SPEAKERS:
        meta = {**DEFAULTS, **KNOWN_SPEAKERS[speaker_id]}
        return meta

    # Priority 3: Auto-detect from patterns
    return {
        "emotion": "neutral",
        "gender": detect_gender(speaker_id, audio_path),
        "age": detect_age(speaker_id),
        "voice_state": "sober",
        "language": "vi",
    }


def annotate_manifest(
    input_path: str,
    output_path: str,
    custom_metadata: Optional[Dict] = None,
) -> dict:
    """Read manifest, add task labels, write extended manifest.

    Returns statistics dict.
    """
    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))

    logger.info(f"Loaded {len(entries)} entries from {input_path}")

    stats = {
        "total": len(entries),
        "gender": Counter(),
        "age": Counter(),
        "emotion": Counter(),
        "voice_state": Counter(),
        "language": Counter(),
        "speakers": Counter(),
    }

    annotated = []
    for entry in entries:
        speaker_id = entry.get("label", "unknown")
        audio_path = entry.get("audio_filepath", "")

        meta = get_speaker_metadata(speaker_id, audio_path, custom_metadata)

        # Add task labels to entry
        entry["emotion"] = meta["emotion"]
        entry["gender"] = meta["gender"]
        entry["age"] = meta["age"]
        entry["voice_state"] = meta["voice_state"]
        entry["language"] = meta["language"]

        annotated.append(entry)

        # Track stats
        for key in ["gender", "age", "emotion", "voice_state", "language"]:
            stats[key][meta[key]] += 1
        stats["speakers"][speaker_id] += 1

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in annotated:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.success(f"Wrote {len(annotated)} annotated entries to {output_path}")
    return stats


def generate_speaker_metadata_template(input_path: str, output_path: str):
    """Generate a template speaker_metadata.json from a manifest.

    User can edit this to correct auto-detected labels.
    """
    speakers = {}
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            sid = entry.get("label", "unknown")
            if sid not in speakers:
                audio_path = entry.get("audio_filepath", "")
                speakers[sid] = {
                    "gender": detect_gender(sid, audio_path),
                    "age": detect_age(sid),
                    "emotion": "neutral",
                    "voice_state": "sober",
                    "language": "vi",
                    "_sample_count": 0,
                }
            speakers[sid]["_sample_count"] += 1

    # Sort by sample count
    speakers = dict(sorted(speakers.items(), key=lambda x: -x[1]["_sample_count"]))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(speakers, f, indent=2, ensure_ascii=False)

    logger.success(
        f"Generated speaker metadata template with {len(speakers)} speakers → {output_path}"
    )
    logger.info("Edit this file to correct auto-detected labels, then re-run with --speaker_metadata")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare multi-task training manifests with paralinguistic labels"
    )
    parser.add_argument("--input_manifest", type=str, required=True,
                        help="Input single-speaker manifest (JSONL)")
    parser.add_argument("--output_manifest", type=str, required=True,
                        help="Output multi-task manifest (JSONL)")
    parser.add_argument("--speaker_metadata", type=str, default=None,
                        help="Custom speaker metadata JSON file")
    parser.add_argument("--generate_template", action="store_true",
                        help="Generate speaker_metadata.json template from manifest")
    parser.add_argument("--val_manifest", type=str, default=None,
                        help="Also annotate validation manifest")
    parser.add_argument("--val_output", type=str, default=None,
                        help="Output path for annotated validation manifest")

    args = parser.parse_args()

    # Generate template mode
    if args.generate_template:
        template_path = args.speaker_metadata or "data/speaker_metadata.json"
        generate_speaker_metadata_template(args.input_manifest, template_path)
        return

    # Load custom metadata if provided
    custom_metadata = None
    if args.speaker_metadata and os.path.exists(args.speaker_metadata):
        with open(args.speaker_metadata, "r") as f:
            custom_metadata = json.load(f)
        # Remove internal fields
        for v in custom_metadata.values():
            v.pop("_sample_count", None)
        logger.info(f"Loaded custom metadata for {len(custom_metadata)} speakers")

    # Annotate training manifest
    stats = annotate_manifest(args.input_manifest, args.output_manifest, custom_metadata)

    # Print statistics
    logger.info("=" * 60)
    logger.info("Multi-Task Dataset Statistics:")
    logger.info(f"  Total samples: {stats['total']}")
    logger.info(f"  Unique speakers: {len(stats['speakers'])}")
    logger.info(f"  Gender distribution: {dict(stats['gender'])}")
    logger.info(f"  Age distribution: {dict(stats['age'])}")
    logger.info(f"  Emotion distribution: {dict(stats['emotion'])}")
    logger.info(f"  Voice state: {dict(stats['voice_state'])}")
    logger.info(f"  Language: {dict(stats['language'])}")
    logger.info("=" * 60)

    # Annotate validation manifest if provided
    if args.val_manifest:
        val_output = args.val_output or args.val_manifest.replace(".json", "_multitask.json")
        annotate_manifest(args.val_manifest, val_output, custom_metadata)


if __name__ == "__main__":
    main()
