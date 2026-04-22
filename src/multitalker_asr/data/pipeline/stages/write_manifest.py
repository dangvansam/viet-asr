"""
WriteManifestStage: write NeMo-compatible JSONL manifest from processed records.
"""

import json
from pathlib import Path
from typing import Dict, List

from loguru import logger

from ..base_stage import BaseStage
from ..checkpoint import PipelineCheckpoint
from ..config import ManifestConfig, PipelineConfig


class WriteManifestStage(BaseStage):
    """Write final NeMo JSONL manifest from all processed records."""

    name = "write_manifest"

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        """
        Write NeMo JSONL manifest from records.
        Skips records without audio_filepath or text.
        Returns records (unchanged — manifest is a side effect).
        """
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / config.manifest.output_filename

        count = 0
        with open(manifest_path, "w", encoding="utf-8") as f:
            for record in records:
                entry = self._build_manifest_entry(record, config.manifest)
                if entry is None:
                    continue
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

        if count == 0:
            logger.warning("Manifest has 0 entries — check text extraction")
        else:
            logger.success(f"Manifest written: {manifest_path} ({count} entries)")

        return records

    def _build_manifest_entry(self, record: Dict, cfg: ManifestConfig) -> Dict:
        """
        Build single NeMo JSONL entry from record.
        Returns None if required fields are missing.
        """
        audio_filepath = record.get("audio_filepath", "")
        if not audio_filepath:
            return None

        # Prefer configured text field, fall back to "text"
        text = record.get(cfg.text_field) or record.get("text", "")
        if not text:
            return None

        entry = {
            "audio_filepath": audio_filepath,
            "text": text,
            "duration": record.get("duration", 0.0),
            "emotion": record.get("emotion", "neutral"),
            "gender": record.get("gender", "unknown"),
            "language": record.get("language", "vi"),
            "textnorm": "withitn" if record.get("text_itn") else "none",
            "speaker_id": record.get("speaker_id", "unknown"),
        }

        # Include alignment only if non-empty
        alignment = record.get("alignment")
        if alignment:
            entry["alignment"] = alignment

        return entry
