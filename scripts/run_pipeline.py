#!/usr/bin/env python3
"""
Multi-task ASR data processing pipeline CLI.

Usage (raw audio/video):
    uv run python scripts/run_pipeline.py \\
        --config configs/pipeline_raw.yaml \\
        --input_dir /data/tiktok_videos \\
        --output_dir /data/processed

Usage (pre-transcribed dataset):
    uv run python scripts/run_pipeline.py \\
        --config configs/pipeline_pretranscribed.yaml \\
        --output_dir /data/processed \\
        --enrich_metadata /home/samdv/DATA/asr/emotion_tongdai/transcripts.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from multitalker_asr.data.pipeline.pipeline import DataPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-task ASR data processing pipeline"
    )
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config")
    parser.add_argument("--input_dir", default=None, help="Directory with audio/video files")
    parser.add_argument("--output_dir", default=None, help="Directory for output files")
    parser.add_argument("--max_files", type=int, default=None, help="Limit number of files to process")
    parser.add_argument(
        "--skip_to_step",
        type=int,
        default=None,
        help="Skip to stage N (1-indexed). All prior stages are skipped.",
    )
    parser.add_argument(
        "--enrich_metadata",
        default=None,
        help="Path to pipe-delimited metadata file (pretranscribed pipeline)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "max_files": args.max_files,
    }
    pipeline = DataPipeline.from_yaml(args.config, **overrides)

    if args.enrich_metadata:
        pipeline._config.enrich.metadata_path = args.enrich_metadata

    if args.skip_to_step:
        pipeline._stages = pipeline._stages[args.skip_to_step - 1:]

    count = pipeline.run()
    print(f"Done. {count} manifest entries written.")


if __name__ == "__main__":
    main()
