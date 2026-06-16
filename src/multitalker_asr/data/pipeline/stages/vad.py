"""
VADStage: standalone voice activity detection that pre-filters near-silent
files and records the speech timeline for downstream consensus checks.
"""

import gc
from typing import Dict, List

from loguru import logger

from ....utils.audio import AudioLoader
from ..base_stage import BaseStage
from ..checkpoint import PipelineCheckpoint
from ..config import PipelineConfig
from ..parallel import parallel_map
from ..vad_backends import build_vad_backend


class VADStage(BaseStage):
    name = "vad"

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
        to_process, done = self._skip_processed(records, checkpoint)
        if not to_process:
            return done

        self._ensure_loaded(config)
        kept: List[Dict] = []
        try:
            def detect(record: Dict):
                try:
                    audio, sr = self._audio_loader.load(record["audio_filepath"])
                    return self._backend.detect(audio, sr)
                except Exception as exc:
                    logger.error(f"VAD failed for {record['id']}: {exc}")
                    return None

            workers = getattr(config, "concurrency", 1) or 1
            results = parallel_map(detect, to_process, workers)
            for record, result in zip(to_process, results):
                record = dict(record)
                checkpoint.mark_processed(record["id"], self.name)
                if result is None:
                    kept.append(record)
                    continue
                if config.vad.enable_prefilter and result.speech_ratio < config.vad.min_speech_ratio:
                    logger.info(
                        f"Drop {record['id']}: speech_ratio "
                        f"{result.speech_ratio:.2f} < {config.vad.min_speech_ratio}"
                    )
                    continue
                record["vad_segments"] = [
                    {"start": seg.start, "end": seg.end} for seg in result.segments
                ]
                record["speech_ratio"] = result.speech_ratio
                kept.append(record)
            checkpoint.save_state()
        finally:
            self._free()

        logger.info(f"VADStage kept {len(kept) - len(done)}/{len(to_process)} new records")
        return done + kept

    def _ensure_loaded(self, config: PipelineConfig) -> None:
        if self._backend is not None:
            return
        kwargs = dict(config.vad.backend_kwargs)
        if config.vad.backend in ("pyannote_seg", "consensus") and config.vad.hf_token:
            kwargs.setdefault("hf_token", config.vad.hf_token)
        self._backend = build_vad_backend(config.vad.backend, **kwargs)
        self._backend.load(device=config.vad.device)

    def _free(self) -> None:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
