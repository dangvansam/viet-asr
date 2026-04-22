"""
EnrichLabelsStage: parse pipe-delimited metadata files into initial records.

Supports two formats:
  - nu-mien-bac (4 fields):  id|speaker|wav_path|text
  - emotion_tongdai (5 fields): id|wav_path|emotion|duration|text
"""

from pathlib import Path
from typing import Dict, List

from loguru import logger

from ..base_stage import BaseStage
from ..checkpoint import PipelineCheckpoint
from ..config import EnrichConfig, PipelineConfig


class EnrichLabelsStage(BaseStage):
    """Parse pipe-delimited metadata into initial record dicts."""

    name = "enrich_labels"

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        """
        Parse pipe-delimited metadata file and return initial records.
        Input 'records' is typically [] — this stage creates records from scratch.
        """
        metadata_path = getattr(config.enrich, "metadata_path", "")
        if not metadata_path:
            logger.warning("enrich_labels: no metadata_path set in config.enrich — returning empty records")
            return records

        parsed = self._parse_metadata_file(metadata_path, config.enrich)
        logger.info(f"EnrichLabels: parsed {len(parsed)} records from {metadata_path}")
        return parsed

    def _parse_metadata_file(
        self,
        metadata_path: str,
        cfg: EnrichConfig,
    ) -> List[Dict]:
        """
        Read metadata_path line by line and return record dicts.
        4 fields → nu-mien-bac format
        5 fields → emotion_tongdai format
        """
        path = Path(metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        records: List[Dict] = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                fields = [field.strip() for field in line.split("|")]
                n = len(fields)

                if n == 4:
                    # nu-mien-bac: id|speaker|wav_path|text
                    id_str, speaker, wav_path, text = fields
                    gender = "female" if "nu" in speaker.lower() else None
                    emotion = "neutral"
                    duration = None

                elif n == 5:
                    # emotion_tongdai: id|wav_path|emotion_raw|duration|text
                    id_str, wav_path, emotion_raw, duration_str, text = fields
                    emotion = cfg.emotion_mapping.get(emotion_raw, "neutral")
                    try:
                        duration = float(duration_str)
                    except ValueError:
                        duration = None
                    gender = None

                else:
                    logger.warning(f"Unexpected field count {n} in line: {line[:80]}")
                    continue

                if not Path(wav_path).exists():
                    logger.warning(f"Audio not found: {wav_path}")

                records.append({
                    "id": self._make_record_id(wav_path),
                    "audio_filepath": str(Path(wav_path).resolve()),
                    "text": text,
                    "text_itn": None,
                    "emotion": emotion,
                    "gender": gender,
                    "language": "vi",
                    "duration": duration,
                })

        return records

    def _make_record_id(self, wav_path: str) -> str:
        """Return Path(wav_path).stem as record id."""
        return Path(wav_path).stem
