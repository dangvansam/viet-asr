#!/usr/bin/env python3
"""
Re-tag the `gender` attribute of an existing manifest using the
gender-classification-service (ensemble wav2vec2 + ECAPA-TDNN), rewriting the
merged manifest and per-platform shards in place.

Usage:
    uv run python scripts/retag_gender.py \
        --manifest /home/samdv/DATA/crawl-2026/manifest_all.jsonl \
        --service-url http://localhost:8000/predict --model ensemble
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import requests
from loguru import logger


class GenderRetagger:
    def __init__(self, service_url: str, model: str = "ensemble", timeout: int = 30):
        self._url = service_url
        self._model = model
        self._timeout = timeout
        self._session = requests.Session()

    def classify(self, audio_path: str):
        with open(audio_path, "rb") as f:
            resp = self._session.post(
                self._url,
                params={"model": self._model, "return_label": False},
                files={"audiofile": f},
                timeout=self._timeout,
            )
        resp.raise_for_status()
        data = resp.json()
        probs = data.get("probs", [0.5, 0.5])
        return data["gender"].lower(), max(probs) if probs else 0.5

    def run(self, manifest_path: str) -> dict:
        path = Path(manifest_path)
        entries = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        stats = {"total": len(entries), "updated": 0, "failed": 0, "flipped": 0}

        for i, e in enumerate(entries):
            audio = e.get("audio_filepath", "")
            if not Path(audio).exists():
                stats["failed"] += 1
                continue
            try:
                gender, conf = self.classify(audio)
            except Exception as exc:
                logger.warning(f"gender service failed for {audio}: {exc}")
                stats["failed"] += 1
                continue
            if e.get("gender") not in (None, gender):
                stats["flipped"] += 1
            e["gender"] = gender
            ac = dict(e.get("attribute_confidence", {}))
            ac["gender"] = conf
            e["attribute_confidence"] = ac
            stats["updated"] += 1
            if (i + 1) % 200 == 0:
                logger.info(f"{i + 1}/{len(entries)} re-tagged")

        self._write(path, entries)
        logger.success(f"Gender re-tag done: {stats}")
        return stats

    def _write(self, merged_path: Path, entries: list) -> None:
        with open(merged_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        shards = defaultdict(list)
        for e in entries:
            shard = str(e.get("extra", {}).get("platform") or "unknown")
            shards[shard].append(e)
        for shard, rows in shards.items():
            shard_path = merged_path.parent / shard / "manifest.jsonl"
            if shard_path.parent.exists():
                with open(shard_path, "w", encoding="utf-8") as f:
                    for e in rows:
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Re-tag gender via the gender service")
    p.add_argument("--manifest", required=True)
    p.add_argument("--service-url", default="http://localhost:8000/predict")
    p.add_argument("--model", default="ensemble")
    args = p.parse_args()
    GenderRetagger(args.service_url, args.model).run(args.manifest)


if __name__ == "__main__":
    main()
