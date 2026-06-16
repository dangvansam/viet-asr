"""
WriteManifestStage: write multitask training manifests (optionally sharded by a
record field, e.g. platform) plus a per-source-item dataset metadata ledger.
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from ..base_stage import BaseStage
from ..checkpoint import PipelineCheckpoint
from ..config import ManifestConfig, PipelineConfig

_EXTRA_KEYS = (
    "platform", "tags", "url", "channel", "data_type", "voice_state",
    "weak_labels", "consensus", "consensus_score", "subtitle_similarity",
    "is_overlap", "speaker_consistency", "relabeled_from", "multitalker_segments",
    "snr_db", "quality_score",
    "cluster_centroid_sim", "cluster_margin", "cluster_threshold",
    "asr_chosen", "asr_agreement", "asr_words_changed", "subtitle_used",
)


class WriteManifestStage(BaseStage):
    """Write multitask JSONL manifest(s) + dataset metadata ledger."""

    name = "write_manifest"

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        entries: List[Dict] = []
        kept_records: List[Dict] = []
        for record in records:
            entry = self._build_manifest_entry(record, config.manifest)
            if entry is None:
                continue
            entries.append(entry)
            kept_records.append(record)

        self._write_manifests(entries, out_dir, config.manifest)

        if config.crawl_seed.seed_path:
            DatasetMetadataWriter(out_dir, config.manifest).write(
                kept_records, config.crawl_seed.seed_path
            )

        return records

    def _write_manifests(
        self, entries: List[Dict], out_dir: Path, cfg: ManifestConfig
    ) -> None:
        if not cfg.shard_by:
            self._write_jsonl(out_dir / cfg.output_filename, entries)
            self._log(len(entries), out_dir / cfg.output_filename)
            return

        shards: Dict[str, List[Dict]] = defaultdict(list)
        for entry in entries:
            shard = str(entry.get(cfg.shard_by) or entry.get("extra", {}).get(cfg.shard_by) or "unknown")
            shards[shard].append(entry)

        for shard, shard_entries in shards.items():
            shard_dir = out_dir / shard
            shard_dir.mkdir(parents=True, exist_ok=True)
            self._write_jsonl(shard_dir / cfg.output_filename, shard_entries)

        merged = out_dir / "manifest_all.jsonl"
        self._write_jsonl(merged, entries)
        self._log(len(entries), merged)

    def _write_jsonl(self, path: Path, entries: List[Dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _log(self, count: int, path: Path) -> None:
        if count == 0:
            logger.warning("Manifest has 0 entries — check text extraction")
        else:
            logger.success(f"Manifest written: {path} ({count} entries)")

    def _build_manifest_entry(self, record: Dict, cfg: ManifestConfig) -> Optional[Dict]:
        audio_filepath = record.get("audio_filepath", "")
        if not audio_filepath:
            return None
        text = record.get(cfg.text_field) or record.get("text", "")
        if not text:
            return None

        num_speakers = int(record.get("num_speakers", 1))
        is_overlap = bool(record.get("extra", {}).get("is_overlap"))
        segment_type = "overlap" if (is_overlap or num_speakers > 1) else "single"

        # Every segment carries all attribute axes: tag/model value or a default.
        attrs = self._fill_attributes(record)
        entry: Dict = {
            "audio_filepath": audio_filepath,
            "offset": record.get("offset", 0.0),
            "duration": record.get("duration", 0.0),
            "text": text,                                  # normalized (ITN + PnC)
            "text_raw": record.get("text_raw", text),      # un-normalized (spoken)
            "num_speakers": num_speakers,
            "speaker_id": record.get("speaker_id", "unknown"),
            "segment_type": segment_type,
            "textnorm": "withitn" if record.get("text_itn") else "none",
            **attrs,
        }
        if record.get("text_itn"):
            entry["text_itn"] = record["text_itn"]
        entry["attribute_confidence"] = self._fill_confidence(record, attrs)
        if record.get("alignment"):
            entry["alignment"] = record["alignment"]
        if record.get("alignment_score") is not None:
            entry["alignment_score"] = record["alignment_score"]
        if record.get("asr_confidence") is not None:
            entry["asr_confidence"] = record["asr_confidence"]

        extra = record.get("extra", {})
        entry_extra = {k: extra[k] for k in _EXTRA_KEYS if extra.get(k) is not None}
        if record.get("consensus_score") is not None:
            entry_extra.setdefault("consensus_score", record["consensus_score"])
        if entry_extra:
            entry["extra"] = entry_extra
        return entry

    def _fill_attributes(self, record: Dict) -> Dict:
        """All 6 axes present; tag/model value or a documented default."""
        return {
            "language": record.get("language") or "vi",
            "emotion": record.get("emotion") or "neutral",
            "gender": record.get("gender") or "unknown",
            "age": record.get("age") or "unknown",
            "region": record.get("region") or "unknown",
            "voice_state": record.get("voice_state") or "sober",
        }

    def _fill_confidence(self, record: Dict, attrs: Dict) -> Dict:
        """attribute_confidence per axis; 0.0 marks a defaulted (unknown) value."""
        conf = dict(record.get("attribute_confidence", {}))
        for axis, value in attrs.items():
            if axis not in conf:
                conf[axis] = 0.0 if value in ("unknown", "neutral", "sober") else 0.5
        return conf


class DatasetMetadataWriter:
    """Per-source-item ledger of processing provenance + outcome."""

    def __init__(self, out_dir: Path, manifest_cfg: ManifestConfig):
        self._out_dir = out_dir
        self._cfg = manifest_cfg

    def write(self, kept_records: List[Dict], seed_path: str) -> None:
        groups: Dict[str, List[Dict]] = defaultdict(list)
        for record in kept_records:
            key = self._source_key(record)
            groups[key].append(record)

        seed_items = self._load_seed(seed_path)
        ledger: List[Dict] = []
        for key, seed in seed_items.items():
            segs = groups.get(key, [])
            ledger.append(self._ledger_line(key, seed, segs))

        self._write_jsonl(self._out_dir / self._cfg.dataset_metadata_filename, ledger)
        summary = self._summarize(ledger)
        (self._out_dir / self._cfg.dataset_summary_filename).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.success(
            f"Dataset ledger: {len(ledger)} source items, "
            f"{summary['n_segments']} segments, {summary['total_kept_hours']}h"
        )

    def _source_key(self, record: Dict) -> str:
        extra = record.get("extra", {})
        return str(extra.get("task_id") or extra.get("url") or record.get("source_audio") or record.get("id"))

    def _load_seed(self, seed_path: str) -> Dict[str, Dict]:
        items: Dict[str, Dict] = {}
        path = Path(seed_path)
        if not path.exists():
            return items
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    seed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = str(seed.get("task_id") or seed.get("url") or seed.get("id"))
                items[key] = seed
        return items

    def _ledger_line(self, key: str, seed: Dict, segs: List[Dict]) -> Dict:
        durations = [float(s.get("duration", 0.0)) for s in segs]
        scores = [s.get("consensus_score") for s in segs if s.get("consensus_score") is not None]
        sims = [
            s.get("extra", {}).get("subtitle_similarity")
            for s in segs
            if s.get("extra", {}).get("subtitle_similarity") is not None
        ]
        return {
            "task_id": seed.get("task_id"),
            "video_id": seed.get("id"),
            "platform": seed.get("platform"),
            "tags": seed.get("tags", []),
            "url": seed.get("url"),
            "channel": seed.get("channel"),
            "data_type": seed.get("data_type"),
            "weak_labels": {
                k: seed[k] for k in ("region", "age", "emotion", "voice_state") if seed.get(k)
            },
            "n_kept_segments": len(segs),
            "kept_duration_s": round(sum(durations), 2),
            "mean_consensus_score": round(sum(scores) / len(scores), 4) if scores else None,
            "mean_subtitle_similarity": round(sum(sims) / len(sims), 4) if sims else None,
            "status": "ok" if segs else "no_kept_segments",
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _summarize(self, ledger: List[Dict]) -> Dict:
        by_platform: Dict[str, int] = defaultdict(int)
        by_data_type: Dict[str, int] = defaultdict(int)
        n_segments = 0
        total_seconds = 0.0
        with_segments = 0
        for line in ledger:
            n_segments += line["n_kept_segments"]
            total_seconds += line["kept_duration_s"]
            if line["n_kept_segments"] > 0:
                with_segments += 1
            by_platform[str(line.get("platform"))] += line["n_kept_segments"]
            by_data_type[str(line.get("data_type"))] += line["n_kept_segments"]
        return {
            "n_source_items": len(ledger),
            "n_items_with_segments": with_segments,
            "n_segments": n_segments,
            "total_kept_hours": round(total_seconds / 3600.0, 3),
            "segments_by_platform": dict(by_platform),
            "segments_by_data_type": dict(by_data_type),
        }

    def _write_jsonl(self, path: Path, rows: List[Dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
