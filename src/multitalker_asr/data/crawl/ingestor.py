"""
CrawlIngestor: sample crawl tasks, download media+subtitle, and emit a seed
manifest (seed.jsonl) the pipeline's CrawlSeedStage consumes.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from .client import CrawlAPIClient
from .downloader import MediaDownloader
from .minio_fetcher import MinIOFetcher
from .sampler import StratifiedSampler
from .tag_taxonomy import CRAWL_TAG_IDS, parse_tag_labels

_WEAK_ATTR_KEYS = ("region", "age", "emotion", "voice_state")


@dataclass
class CrawlIngestConfig:
    out_root: str
    platforms: List[str] = field(default_factory=lambda: ["tiktok", "youtube", "facebook"])
    tag_ids: List[int] = field(default_factory=lambda: list(CRAWL_TAG_IDS))
    total: int = 1000
    seed: int = 42
    download: bool = True
    vietnamese: bool = True
    status: str = "completed"
    subtitle_langs: Optional[List[str]] = None
    page_limit: int = 1000
    floor: int = 1
    api_url: Optional[str] = None
    api_token: Optional[str] = None
    use_minio: bool = True


class CrawlIngestor:
    def __init__(self, config: CrawlIngestConfig):
        self._cfg = config
        self._client = CrawlAPIClient(base_url=config.api_url, token=config.api_token)
        self._sampler = StratifiedSampler(page_limit=config.page_limit, floor=config.floor)
        minio_fetcher = MinIOFetcher.from_env() if config.use_minio else None
        self._downloader = MediaDownloader(
            self._client,
            config.out_root,
            subtitle_langs=config.subtitle_langs,
            minio_fetcher=minio_fetcher,
        )

    def run(self) -> Dict:
        result = self._sampler.sample(
            self._client,
            platforms=self._cfg.platforms,
            tag_ids=self._cfg.tag_ids,
            total=self._cfg.total,
            seed=self._cfg.seed,
            status=self._cfg.status,
        )

        seed_records: List[Dict] = []
        failures = 0
        for task in result.tasks:
            dl = self._downloader.download(task, download_media=self._cfg.download)
            if not dl["ok"]:
                failures += 1
            record = self._build_seed_record(task, dl)
            if record is not None:
                seed_records.append(record)

        seed_path = self._write_seed(seed_records)
        self._write_distribution(result.realized, len(seed_records), failures)

        summary = {
            "sampled": len(result.tasks),
            "seed_records": len(seed_records),
            "download_failures": failures,
            "seed_path": str(seed_path),
            "downloaded": self._cfg.download,
        }
        logger.success(f"Ingest complete: {summary}")
        return summary

    def _build_seed_record(self, task: Dict, dl: Dict) -> Optional[Dict]:
        tag_names = [t.get("name", "") for t in task.get("tags", [])]
        weak = parse_tag_labels(tag_names)
        ident = self._downloader._identifier(task)

        audio_filepath = dl.get("audio_path")
        if audio_filepath is None:
            audio_filepath = str(self._downloader.item_dir(task) / "audio.wav")

        record: Dict = {
            "id": ident,
            "audio_filepath": audio_filepath,
            "subtitle_path": dl.get("subtitle_path"),
            "platform": task.get("platform"),
            "tags": tag_names,
            "url": task.get("url"),
            "channel": task.get("channel_name"),
            "task_id": task.get("task_id"),
            "data_type": weak.get("data_type"),
            "media_kind": dl.get("media_kind"),
        }
        if self._cfg.vietnamese:
            record["language"] = "vi"
        for key in _WEAK_ATTR_KEYS:
            if key in weak:
                record[key] = weak[key]
        return record

    def _write_seed(self, records: List[Dict]) -> Path:
        seed_dir = Path(self._cfg.out_root) / "_seed"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_path = seed_dir / "seed.jsonl"
        with open(seed_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Wrote seed manifest: {seed_path} ({len(records)} records)")
        return seed_path

    def _write_distribution(self, realized: Dict[str, int], n_records: int, failures: int) -> None:
        seed_dir = Path(self._cfg.out_root) / "_seed"
        seed_dir.mkdir(parents=True, exist_ok=True)
        by_platform: Dict[str, int] = {}
        for key, n in realized.items():
            platform = key.split(":", 1)[0]
            by_platform[platform] = by_platform.get(platform, 0) + n
        dist = {
            "by_cell": realized,
            "by_platform": by_platform,
            "seed_records": n_records,
            "download_failures": failures,
            "config": {
                "platforms": self._cfg.platforms,
                "tag_ids": self._cfg.tag_ids,
                "total": self._cfg.total,
                "seed": self._cfg.seed,
            },
        }
        (seed_dir / "sample_distribution.json").write_text(
            json.dumps(dist, ensure_ascii=False, indent=2), encoding="utf-8"
        )
