"""
AlignStage: word-level forced alignment via Qwen3ForcedAligner.
Adapted from /home/samdv/data-processing-pipeline/pipeline/alignment.py.
"""

import gc
from typing import Dict, List, Optional

from loguru import logger

from ..base_stage import BaseStage
from ..checkpoint import PipelineCheckpoint
from ..config import AlignConfig, PipelineConfig


class AlignStage(BaseStage):
    """Run Qwen3ForcedAligner to produce word-level timestamps."""

    name = "align"

    def __init__(self) -> None:
        self._aligner = None

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        """
        Run Qwen3ForcedAligner on each record.
        Uses record["text_itn"] as input text; falls back to record["text"].
        Frees GPU after all records processed.
        """
        to_process, done = self._skip_processed(records, checkpoint)
        if not to_process:
            return done

        self._load_aligner(config.align)
        processed: List[Dict] = list(done)

        try:
            for record in to_process:
                record = dict(record)
                text = record.get("text_itn") or record.get("text", "")

                if not text:
                    logger.warning(f"No text for alignment: {record['id']}")
                    record["alignment"] = []
                    record["alignment_score"] = 0.0
                    processed.append(record)
                    checkpoint.mark_processed(record["id"], self.name)
                    checkpoint.save_state()
                    continue

                try:
                    alignment = self._align_one(
                        record["audio_filepath"], text, config.align.language
                    )
                    record["alignment"] = alignment
                    record["alignment_score"] = self._calculate_alignment_score(alignment)
                except Exception as e:
                    logger.error(f"Alignment failed for {record['id']}: {e}")
                    record["alignment"] = []
                    record["alignment_score"] = 0.0

                checkpoint.mark_processed(record["id"], self.name)
                checkpoint.save_state()
                processed.append(record)

        finally:
            self._free_gpu()

        return processed

    def _load_aligner(self, cfg: AlignConfig) -> None:
        """Load Qwen3ForcedAligner model. Cached on self._aligner."""
        if self._aligner is not None:
            return
        try:
            from qwen_asr import Qwen3ForcedAligner
        except ImportError:
            raise ImportError("qwen_asr not installed — run: pip install qwen-asr")

        import torch
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)
        device_map = f"{cfg.device}:0" if "cuda" in cfg.device else cfg.device

        logger.info(f"Loading Qwen3ForcedAligner: {cfg.model}")
        self._aligner = Qwen3ForcedAligner.from_pretrained(
            cfg.model,
            dtype=torch_dtype,
            device_map=device_map,
        )
        logger.info("Qwen3ForcedAligner loaded")

    def _align_one(
        self,
        audio_path: str,
        text: str,
        language: str,
    ) -> List[Dict]:
        """
        Run alignment for a single audio+text pair.
        Returns list of {"text": str, "start_time": float, "end_time": float}.
        Returns empty list on failure (does not raise).
        """
        try:
            results = self._aligner.align(
                audio=[audio_path],
                text=[text],
                language=[language],
            )
            if results and len(results) > 0 and len(results[0]) > 0:
                return [
                    {
                        "text": item.text,
                        "start_time": item.start_time,
                        "end_time": item.end_time,
                    }
                    for item in results[0]
                ]
            logger.warning(f"Empty alignment for {audio_path}")
            return []
        except Exception as e:
            logger.error(f"Alignment error for {audio_path}: {e}")
            return []

    def _calculate_alignment_score(self, alignment: List[Dict]) -> float:
        """
        Fraction of segments with valid timestamps (start >= 0 and end > start).
        Returns 0.0 for empty alignment.
        """
        if not alignment:
            return 0.0
        valid = sum(
            1
            for seg in alignment
            if seg.get("start_time", -1) >= 0 and seg.get("end_time", -1) > seg.get("start_time", -1)
        )
        return valid / len(alignment)

    def _free_gpu(self) -> None:
        """Free GPU memory from aligner."""
        if self._aligner is not None:
            del self._aligner
            self._aligner = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
