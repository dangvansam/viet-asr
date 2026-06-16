"""
AlignStage: word-level forced alignment via a pluggable align backend.
Input text is the ensemble transcript produced upstream (text_itn / text).
"""

import gc
from typing import Dict, List

from loguru import logger

from ..align_backends import build_align_backend
from ..base_stage import BaseStage
from ..checkpoint import PipelineCheckpoint
from ..config import AlignConfig, PipelineConfig
from ..segment_utils import speech_span


class AlignStage(BaseStage):
    """Run a forced aligner to produce word-level timestamps."""

    name = "align"

    def __init__(self) -> None:
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

        self._ensure_loaded(config.align)
        processed: List[Dict] = list(done)

        try:
            for record in to_process:
                record = dict(record)
                text = record.get(config.align.text_field) or record.get("text", "")

                if not text:
                    logger.warning(f"No text for alignment: {record['id']}")
                    record["alignment"] = []
                    record["alignment_score"] = 0.0
                else:
                    record = self._align_record(record, text, config.align)

                checkpoint.mark_processed(record["id"], self.name)
                checkpoint.save_state()
                processed.append(record)
        finally:
            self._free_gpu()

        return processed

    def _align_record(self, record: Dict, text: str, cfg: AlignConfig) -> Dict:
        try:
            result = self._backend.align(record["audio_filepath"], text, cfg.language)
        except Exception as exc:
            logger.error(f"Alignment failed for {record['id']}: {exc}")
            record["alignment"] = []
            record["alignment_score"] = 0.0
            return record

        record["alignment"] = result.to_records()
        record["alignment_score"] = result.score

        if cfg.trim_to_words and result.span is not None and result.score > 0:
            self._trim_to_words(record, result.span, cfg.trim_pad_s)
        return record

    def _trim_to_words(self, record: Dict, span, pad_s: float) -> None:
        duration = float(record.get("duration", 0.0))
        bounded = speech_span([span], pad_s=pad_s, lo=0.0, hi=duration or None)
        if bounded is None:
            return
        record.setdefault("extra", {})["align_trim"] = {
            "original_offset": float(record.get("offset", 0.0)),
            "original_duration": duration,
            "start": bounded[0],
            "end": bounded[1],
        }
        record["offset"] = float(record.get("offset", 0.0)) + bounded[0]
        record["duration"] = bounded[1] - bounded[0]

    def _ensure_loaded(self, cfg: AlignConfig) -> None:
        if self._backend is not None:
            return
        kwargs = dict(cfg.backend_kwargs)
        if cfg.backend == "qwen3":
            kwargs.setdefault("model", cfg.model)
            kwargs.setdefault("dtype", cfg.dtype)
        logger.info(f"Loading align backend: {cfg.backend}")
        self._backend = build_align_backend(cfg.backend, **kwargs)
        self._backend.load(device=cfg.device)

    def _free_gpu(self) -> None:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
