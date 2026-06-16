"""
MultitalkerReprocessStage: re-transcribe overlap-flagged segments with the
project's multitalker ASR model (joint Sortformer diarization + multi-speaker
RNNT) to recover per-speaker transcripts for simultaneous speech, instead of
dropping them.

Heavy: loads a .nemo multitalker model + Sortformer (GPU recommended). Disabled
by default (config.overlap_reprocess.enabled). Non-overlap segments pass through.
"""

from typing import Dict, List, Optional

from loguru import logger

from ..base_stage import BaseStage
from ..checkpoint import PipelineCheckpoint
from ..config import OverlapReprocessConfig, PipelineConfig


class MultitalkerReprocessStage(BaseStage):
    name = "multitalker_reprocess"

    def __init__(self) -> None:
        self._transcriber = None

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        cfg = config.overlap_reprocess
        if not cfg.enabled:
            return records

        overlap = [r for r in records if r.get("extra", {}).get("is_overlap")]
        if not overlap:
            logger.info("MultitalkerReprocess: no overlap segments, skipping")
            return records

        try:
            self._load(cfg)
        except Exception as exc:
            logger.error(f"MultitalkerReprocess: model load failed ({exc}); skipping")
            return records

        out: List[Dict] = []
        reprocessed = 0
        for record in records:
            if not record.get("extra", {}).get("is_overlap"):
                out.append(record)
                continue
            record = dict(record)
            spk_segments = self._transcribe(record["audio_filepath"])
            if spk_segments:
                extra = dict(record.get("extra") or {})
                extra["multitalker_segments"] = spk_segments
                record["extra"] = extra
                record["num_speakers"] = len({s.get("speaker") for s in spk_segments})
                primary = max(spk_segments, key=lambda s: len(s.get("text", "")))
                record["text"] = primary.get("text", record.get("text", ""))
                record.setdefault("text_itn", record["text"])
                reprocessed += 1
            checkpoint.mark_processed(record["id"], self.name)
            out.append(record)

        checkpoint.save_state()
        logger.info(f"MultitalkerReprocess: reprocessed {reprocessed} overlap segments")
        self._free()
        return out

    def _transcribe(self, audio_path: str) -> List[Dict]:
        try:
            results = self._transcriber.transcribe(audio_path, output_path=None)
        except Exception as exc:
            logger.warning(f"multitalker transcribe failed for {audio_path}: {exc}")
            return []
        if not results:
            return []
        segments = results[0].get("segments") if isinstance(results[0], dict) else None
        if not segments:
            return []
        return [
            {
                "speaker": str(s.get("speaker", "")),
                "text": s.get("text", ""),
                "start": float(s.get("start_time", 0.0)),
                "end": float(s.get("end_time", 0.0)),
            }
            for s in segments
        ]

    def _load(self, cfg: OverlapReprocessConfig) -> None:
        if self._transcriber is not None:
            return
        from ....configs import ModelConfig
        from ....inference import Transcriber
        from ....models import MultitalkerASRModel

        logger.info(f"Loading multitalker model: {cfg.asr_model_path}")
        model_cfg = ModelConfig(
            asr_model_path=cfg.asr_model_path,
            diar_model_path=cfg.diar_model_path,
        )
        model = MultitalkerASRModel(model_cfg)
        model.load_models()
        self._transcriber = Transcriber(model)

    def _free(self) -> None:
        self._transcriber = None
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
