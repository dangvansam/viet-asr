"""
AudioQualityStage: estimate audio SNR before VAD and drop very noisy files early
(music/ambient noise breaks VAD + diarization). Lightweight numpy estimator by
default (no model); pluggable via quality_backends.
"""

from typing import Dict, List

from loguru import logger

from ....utils.audio import AudioLoader
from ..base_stage import BaseStage
from ..checkpoint import PipelineCheckpoint
from ..config import PipelineConfig
from ..quality_backends import build_quality_backend


class AudioQualityStage(BaseStage):
    name = "audio_quality"

    def __init__(self, sample_rate: int = 16000) -> None:
        self._sample_rate = sample_rate
        self._audio_loader = AudioLoader(target_sample_rate=sample_rate)
        self._backend = None

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        cfg = config.audio_quality
        if not cfg.enabled:
            return records

        to_process, done = self._skip_processed(records, checkpoint)
        if not to_process:
            return done

        self._ensure_loaded(cfg)
        kept: List[Dict] = []
        dropped = 0
        for record in to_process:
            record = dict(record)
            try:
                audio, sr = self._audio_loader.load(record["audio_filepath"])
                result = self._backend.score(audio, sr)
            except Exception as exc:
                logger.warning(f"SNR estimate failed for {record['id']}: {exc} — keeping")
                checkpoint.mark_processed(record["id"], self.name)
                checkpoint.save_state()
                kept.append(record)
                continue

            extra = dict(record.get("extra") or {})
            extra["snr_db"] = result.snr_db
            extra["quality_score"] = result.score
            record["extra"] = extra
            checkpoint.mark_processed(record["id"], self.name)
            checkpoint.save_state()

            if result.snr_db < cfg.min_snr_db:
                dropped += 1
                logger.info(f"Drop {record['id']}: SNR {result.snr_db:.1f}dB < {cfg.min_snr_db}")
                continue
            kept.append(record)

        logger.info(f"AudioQuality kept {len(kept)}/{len(to_process)} (dropped {dropped} noisy)")
        return done + kept

    def _ensure_loaded(self, cfg) -> None:
        if self._backend is not None:
            return
        self._backend = build_quality_backend(cfg.backend, **dict(cfg.backend_kwargs))
        self._backend.load(device=cfg.device)
