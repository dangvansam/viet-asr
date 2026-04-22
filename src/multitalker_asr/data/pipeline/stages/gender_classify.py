"""
GenderClassifyStage: classify gender of audio segments via HTTP API.
"""

from typing import Dict, List

from loguru import logger

from ..base_stage import BaseStage
from ..checkpoint import PipelineCheckpoint
from ..config import GenderConfig, PipelineConfig

_SAVE_EVERY = 50  # save checkpoint every N records


class GenderClassifyStage(BaseStage):
    """Classify gender via HTTP POST to gender-classification-service."""

    name = "gender_classify"

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        """
        Classify gender for each audio segment via HTTP API.
        Skips records already processed (checkpoint).
        Skips records where gender is already set.
        If service unavailable, returns records unchanged.
        """
        cfg = config.gender

        if not self._check_service(cfg.url):
            logger.warning(
                f"Gender service unavailable at {cfg.url}, skipping gender classification"
            )
            return records

        to_process, done = self._skip_processed(records, checkpoint)

        processed: List[Dict] = list(done)
        for i, record in enumerate(to_process):
            record = dict(record)

            # Skip if gender already set (e.g. from enrich_labels)
            if record.get("gender"):
                processed.append(record)
                continue

            audio_path = record.get("audio_filepath", "")
            try:
                result = self._classify_one(audio_path, cfg)
                record["gender"] = result["gender"]
                record["gender_confidence"] = result["gender_confidence"]
                checkpoint.mark_processed(record["id"], self.name)
            except Exception as e:
                logger.error(f"Gender API error for {record['id']}: {e}")
                record["gender"] = None
                record["gender_confidence"] = None

            if (i + 1) % _SAVE_EVERY == 0:
                checkpoint.save_state()

            processed.append(record)

        checkpoint.save_state()
        return processed

    def _classify_one(self, audio_path: str, cfg: GenderConfig) -> Dict:
        """
        POST audio to gender service.
        Returns {"gender": "male"|"female", "gender_confidence": float}.
        """
        import requests

        with open(audio_path, "rb") as f:
            resp = requests.post(
                cfg.url,
                params={"model": cfg.model, "return_label": False},
                files={"audiofile": f},
                timeout=cfg.timeout,
            )
        resp.raise_for_status()
        data = resp.json()
        gender = data["gender"].lower()  # "MALE" → "male"
        probs = data.get("probs", [0.5, 0.5])
        confidence = max(probs) if probs else 0.5
        return {"gender": gender, "gender_confidence": confidence}

    def _check_service(self, url: str, timeout: int = 5) -> bool:
        """GET {url}/health_check → True if 200, False otherwise."""
        import requests

        try:
            resp = requests.get(f"{url}/health_check", timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False
