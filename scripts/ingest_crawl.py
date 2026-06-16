#!/usr/bin/env python3
"""
Ingest social-video-crawl data into a training-ready folder.

Samples completed crawl tasks (stratified over platform x STT/TTS tag), downloads
audio + Vietnamese subtitle into {out_root}/{platform}/{primary_tag}/{video_id}/,
and writes a seed manifest the pipeline consumes.

Usage:
    uv run python scripts/ingest_crawl.py \
        --sample 1000 --platforms tiktok,youtube,facebook \
        --tag-family stt,tts --vietnamese --seed 42 \
        --out-root /home/samdv/DATA/crawl-2026

Token resolution: --api-token, else $CRAWL_API_TOKEN, else API_TOKEN in
/home/samdv/social-video-crawl/.env.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from multitalker_asr.data.crawl import (  # noqa: E402
    CrawlIngestConfig,
    CrawlIngestor,
    resolve_tag_family,
)

_CRAWL_ENV = "/home/samdv/social-video-crawl/.env"


def resolve_token(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if os.environ.get("CRAWL_API_TOKEN"):
        return os.environ["CRAWL_API_TOKEN"]
    env_path = Path(_CRAWL_ENV)
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("API_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest social-video-crawl data")
    p.add_argument("--sample", type=int, default=1000, help="Total items to sample")
    p.add_argument("--platforms", default="tiktok,youtube,facebook")
    p.add_argument("--tag-family", default="stt,tts", help="Comma list: stt,tts")
    p.add_argument("--vietnamese", action="store_true", help="Seed language=vi")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-root", default="/home/samdv/DATA/crawl-2026")
    p.add_argument("--api-url", default=os.environ.get("CRAWL_API_URL"))
    p.add_argument("--api-token", default=None)
    p.add_argument("--floor", type=int, default=1, help="Min items per (platform,tag) cell")
    p.add_argument("--no-download", action="store_true", help="Dry-run: sample + seed only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = CrawlIngestConfig(
        out_root=args.out_root,
        platforms=[p.strip() for p in args.platforms.split(",") if p.strip()],
        tag_ids=resolve_tag_family([f for f in args.tag_family.split(",") if f.strip()]),
        total=args.sample,
        seed=args.seed,
        download=not args.no_download,
        vietnamese=args.vietnamese,
        floor=args.floor,
        api_url=args.api_url,
        api_token=resolve_token(args.api_token),
    )
    summary = CrawlIngestor(config).run()
    print(summary)


if __name__ == "__main__":
    main()
