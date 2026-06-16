#!/usr/bin/env python3
"""
Stream-process social-video-crawl tasks on the fly (disk-bounded, parallel).

Pulls completed tasks from the crawl DB (API), downloads each by task_id,
runs the pipeline with a pool of GPU workers, writes a growing manifest, and
deletes raw + intermediate audio — keeping only segments/ + manifest + meta.
Re-running resumes from where it stopped (per-task_id checkpoint).

Usage:
    uv run python scripts/stream_crawl.py --config configs/pipeline_crawl_stream.yaml \
        [--platforms tiktok,youtube] [--tag-family stt,tts] [--max-items 500] \
        [--batch-size 32] [--gpu-workers 3] [--download-workers 16] \
        [--devices cuda:1] [--output_dir /home/samdv/DATA/crawl-2026]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from multitalker_asr.data.pipeline.config import PipelineConfig  # noqa: E402
from multitalker_asr.data.pipeline.stream_runner import StreamPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream-process crawl tasks on the fly")
    p.add_argument("--config", required=True, help="Pipeline + stream YAML config")
    p.add_argument("--source-backend", choices=["api", "db"], default=None,
                   help="Override crawl_source.backend (api=HTTP :8010, db=Postgres)")
    p.add_argument("--platforms", default=None, help="Comma list override")
    p.add_argument("--tag-family", default=None, help="Comma list: stt,tts")
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--gpu-workers", type=int, default=None)
    p.add_argument("--download-workers", type=int, default=None)
    p.add_argument("--devices", default=None, help="Comma list, e.g. cuda:1,cuda:1")
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def apply_overrides(config: PipelineConfig, args: argparse.Namespace) -> None:
    if args.output_dir:
        config.output_dir = args.output_dir
        config.__post_init__()
    if args.source_backend:
        config.crawl_source.backend = args.source_backend
    if args.platforms:
        config.crawl_source.platforms = [p.strip() for p in args.platforms.split(",") if p.strip()]
    if args.tag_family:
        config.crawl_source.tag_family = [t.strip() for t in args.tag_family.split(",") if t.strip()]
    if args.max_items is not None:
        config.crawl_source.max_items = args.max_items
    if args.batch_size is not None:
        config.stream.batch_size = args.batch_size
    if args.gpu_workers is not None:
        config.stream.gpu_workers = args.gpu_workers
    if args.download_workers is not None:
        config.stream.download_workers = args.download_workers
    if args.devices:
        config.stream.devices = [d.strip() for d in args.devices.split(",") if d.strip()]


def main() -> None:
    args = parse_args()
    config = PipelineConfig.from_yaml(args.config)
    apply_overrides(config, args)
    summary = StreamPipeline(config).run()
    print(summary)


if __name__ == "__main__":
    main()
