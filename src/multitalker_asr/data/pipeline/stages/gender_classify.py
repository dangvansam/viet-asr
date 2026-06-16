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
                # Stamp attribute_confidence so FilterStage won't mask gender.
                conf = dict(record.get("attribute_confidence", {}))
                conf["gender"] = result["gender_confidence"]
                record["attribute_confidence"] = conf
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
        POST audio to the gender service `/v1/audio/classifications` (OpenAI-style).
        Falls back to the legacy `/predict` (404) for zero-downtime rollout.
        Returns {"gender": "male"|"female", "gender_confidence": float}.
        """
        import requests

        base = self._base_url(cfg.url)
        with open(audio_path, "rb") as f:
            resp = requests.post(
                f"{base}/v1/audio/classifications",
                data={"model": cfg.model, "probs": "true"},
                files={"file": f},
                timeout=cfg.timeout,
            )
        if resp.status_code == 404:
            with open(audio_path, "rb") as f:
                resp = requests.post(
                    f"{base}/predict",
                    params={"model": cfg.model, "return_label": False},
                    files={"audiofile": f},
                    timeout=cfg.timeout,
                )
        resp.raise_for_status()
        return self._parse(resp.json())

    @staticmethod
    def _parse(data: Dict) -> Dict:
        """Parse the OpenAI-style classification (tolerant of the legacy /predict shape)."""
        label = (data.get("label") or data.get("gender") or "").lower()
        if data.get("confidence") is not None:
            confidence = float(data["confidence"])
        else:
            probs = data.get("probs", [0.5, 0.5])
            if isinstance(probs, dict):
                confidence = max(probs.values()) if probs else 0.5
            else:
                confidence = max(probs) if probs else 0.5
        return {"gender": label, "gender_confidence": confidence}

    @staticmethod
    def _base_url(url: str) -> str:
        base = url.rstrip("/")
        for suffix in ("/v1/audio/classifications", "/predict"):
            if base.endswith(suffix):
                return base[: -len(suffix)]
        return base

    def _check_service(self, url: str, timeout: int = 5) -> bool:
        """Probe the service root → True if 200. Tries /health then /health_check then /."""
        import requests

        base = self._base_url(url)
        for path in ("/health", "/health_check", "/"):
            try:
                resp = requests.get(f"{base}{path}", timeout=timeout)
                if resp.status_code == 200:
                    return True
            except Exception:
                continue
        return False
