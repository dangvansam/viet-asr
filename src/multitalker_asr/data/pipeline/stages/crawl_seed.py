"""
CrawlSeedStage: read the seed.jsonl produced by scripts/ingest_crawl.py and
emit initial pipeline records carrying tag-derived weak labels + provenance.
"""

import json
from pathlib import Path
from typing import Dict, List

from loguru import logger

from ..base_stage import BaseStage
from ..checkpoint import PipelineCheckpoint
from ..config import PipelineConfig

_WEAK_ATTR_KEYS = ("region", "age", "emotion", "voice_state")
_ATTRIBUTE_KEYS = ("region", "age", "emotion")


class CrawlSeedStage(BaseStage):
    """Generate initial records from a crawl seed manifest (created from scratch)."""

    name = "crawl_seed"

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        seed_path = config.crawl_seed.seed_path
        if not seed_path:
            logger.warning("crawl_seed: no seed_path configured — returning input records")
            return records

        path = Path(seed_path)
        if not path.exists():
            raise FileNotFoundError(f"Seed manifest not found: {seed_path}")

        built: List[Dict] = []
        skipped = 0
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    seed = json.loads(line)
                except json.JSONDecodeError as exc:
                    skipped += 1
                    logger.warning(f"CrawlSeed: skip malformed seed line {lineno}: {exc}")
                    continue
                built.append(self._build_record(seed))

        max_files = getattr(config, "max_files", None)
        if max_files:
            built = built[:max_files]

        msg = f"CrawlSeed: built {len(built)} records from {seed_path}"
        if skipped:
            msg += f" ({skipped} malformed lines skipped)"
        if max_files:
            msg += f" (capped to max_files={max_files})"
        logger.info(msg)
        return built

    def _build_record(self, seed: Dict) -> Dict:
        weak = {k: seed[k] for k in _WEAK_ATTR_KEYS if seed.get(k)}
        record: Dict = {
            "id": seed["id"],
            "audio_filepath": seed["audio_filepath"],
        }
        confidence: Dict[str, float] = {}
        if seed.get("language"):
            record["language"] = seed["language"]
            confidence["language"] = 0.99
        for key, value in weak.items():
            record[key] = value
            if key in _ATTRIBUTE_KEYS:
                confidence[key] = 0.95
        if confidence:
            record["attribute_confidence"] = confidence

        record["extra"] = {
            "platform": seed.get("platform"),
            "tags": seed.get("tags", []),
            "url": seed.get("url"),
            "channel": seed.get("channel"),
            "task_id": seed.get("task_id"),
            "data_type": seed.get("data_type"),
            "media_kind": seed.get("media_kind"),
            "subtitle_path": seed.get("subtitle_path"),
            "weak_labels": weak,
        }
        record["platform"] = seed.get("platform")
        return record
