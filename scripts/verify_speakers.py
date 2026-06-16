#!/usr/bin/env python3
"""
QC an existing manifest with the speaker-recognition /embed service: per source
file, re-cluster segment embeddings to verify/correct speaker_id labels, drop
ambiguous segments, set num_speakers, and rewrite the merged + sharded manifests.

Usage:
    uv run python scripts/verify_speakers.py \
        --manifest /home/samdv/DATA/crawl-2026/manifest_all.jsonl \
        --service-url http://localhost:2010/embed \
        --cluster-threshold 0.45 --ambiguous-max 0.30
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from multitalker_asr.data.pipeline.speaker import (  # noqa: E402
    SpeakerEmbedder,
    centroids,
    cluster_embeddings,
    consistency_scores,
    cosine,
)

_SOURCE_RE = re.compile(r"^(.*?)_(?:SPEAKER_\d+|SPK_\d+|SEG_\d+)_")


def source_key(entry: dict) -> str:
    seg_id = entry.get("id") or Path(entry.get("audio_filepath", "")).stem
    m = _SOURCE_RE.match(seg_id)
    if m:
        return m.group(1)
    return entry.get("extra", {}).get("url") or seg_id


class SpeakerVerifier:
    def __init__(self, url, cluster_threshold, ambiguous_max, drop_ambiguous, min_cluster_size):
        self._embedder = SpeakerEmbedder(url)
        self._threshold = cluster_threshold
        self._ambiguous_max = ambiguous_max
        self._drop = drop_ambiguous
        self._min_cluster = min_cluster_size

    def run(self, manifest_path: str) -> dict:
        path = Path(manifest_path)
        entries = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not self._embedder.health():
            raise RuntimeError("speaker service not reachable")

        groups = defaultdict(list)
        for e in entries:
            groups[source_key(e)].append(e)

        kept, stats = [], {
            "files": len(groups), "segments": len(entries),
            "relabeled": 0, "dropped": 0, "embed_fail": 0,
        }
        for _, segs in groups.items():
            kept.extend(self._verify(segs, stats))

        self._write(path, kept)
        logger.success(f"verify_speakers done: {stats} -> {len(kept)} kept")
        return stats

    def _verify(self, segs, stats):
        embs, valid, passthrough = [], [], []
        for s in segs:
            v = self._embedder.embed(s["audio_filepath"])
            if v is None:
                stats["embed_fail"] += 1
                passthrough.append(s)
            else:
                embs.append(v)
                valid.append(s)
        if len(valid) < self._min_cluster:
            for s in valid:
                s["num_speakers"] = len({x.get("speaker_id") for x in valid}) or 1
            return passthrough + valid

        matrix = np.stack(embs)
        labels = cluster_embeddings(matrix, self._threshold)
        cents = centroids(matrix, labels)
        scores = consistency_scores(matrix, labels)
        num_spk = len(set(labels.tolist()))

        out = list(passthrough)
        for s, lab, score, emb in zip(valid, labels, scores, embs):
            s["num_speakers"] = num_spk
            s.setdefault("extra", {})["speaker_consistency"] = round(float(score), 4)
            new_spk = f"SPK_{int(lab)}"
            if new_spk != s.get("speaker_id"):
                stats["relabeled"] += 1
                s["extra"]["relabeled_from"] = s.get("speaker_id")
                s["speaker_id"] = new_spk
            is_overlap = bool(s.get("extra", {}).get("is_overlap", False))
            if self._drop and not is_overlap and score < self._ambiguous_max \
                    and cosine(emb, cents[int(lab)]) < self._ambiguous_max:
                stats["dropped"] += 1
                continue
            out.append(s)
        return out

    def _write(self, merged: Path, entries):
        with open(merged, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        shards = defaultdict(list)
        for e in entries:
            shards[str(e.get("extra", {}).get("platform") or "unknown")].append(e)
        for shard, rows in shards.items():
            sp = merged.parent / shard / "manifest.jsonl"
            if sp.parent.exists():
                with open(sp, "w", encoding="utf-8") as f:
                    for e in rows:
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description="QC speaker labels on a manifest")
    p.add_argument("--manifest", required=True)
    p.add_argument("--service-url", default="http://localhost:2010/embed")
    p.add_argument("--cluster-threshold", type=float, default=0.45)
    p.add_argument("--ambiguous-max", type=float, default=0.30)
    p.add_argument("--min-cluster-size", type=int, default=2)
    p.add_argument("--no-drop", action="store_true")
    args = p.parse_args()

    backup = Path(args.manifest).with_suffix(".spk_backup.jsonl")
    if not backup.exists():
        backup.write_text(Path(args.manifest).read_text(encoding="utf-8"), encoding="utf-8")
        logger.info(f"backup: {backup}")

    SpeakerVerifier(
        args.service_url, args.cluster_threshold, args.ambiguous_max,
        not args.no_drop, args.min_cluster_size,
    ).run(args.manifest)


if __name__ == "__main__":
    main()
