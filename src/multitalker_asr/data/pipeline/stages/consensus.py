"""
ConsensusStage: cross-signal agreement gate. Keeps a segment as training data
only when VAD/diarization-derived bounds, the forced-alignment span, the ASR
ensemble confidence, and (for crawl data) the subtitle text + tag-derived weak
labels agree within tolerance.
"""

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from ..base_stage import BaseStage
from ..checkpoint import PipelineCheckpoint
from ..config import ConsensusConfig, PipelineConfig
from ..segment_utils import interval_iou


class ConsensusStage(BaseStage):
    name = "consensus"

    def __init__(self) -> None:
        self._vtt_parser = None
        self._cue_cache: Dict[str, list] = {}

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        cfg = config.consensus
        kept: List[Dict] = []
        dropped = 0

        for record in records:
            record = dict(record)
            checks = self._evaluate(record, cfg)
            score = sum(1 for v in checks.values() if v) / len(checks) if checks else 0.0
            record["consensus"] = checks
            record["consensus_score"] = score
            record.setdefault("extra", {})["consensus"] = checks

            if cfg.drop_on_disagreement and self._hard_fail(checks, cfg):
                dropped += 1
                logger.debug(f"Drop {record.get('id')} via consensus: {checks}")
                continue

            checkpoint.mark_processed(record["id"], self.name)
            kept.append(record)

        logger.info(
            f"ConsensusStage kept {len(kept)}/{len(records)} records (dropped {dropped})"
        )
        return kept

    def _evaluate(self, record: Dict, cfg: ConsensusConfig) -> Dict[str, bool]:
        checks: Dict[str, bool] = {}
        duration = float(record.get("duration", 0.0))
        alignment = record.get("alignment") or []

        if duration > 0 and alignment:
            span = (float(alignment[0]["start_time"]), float(alignment[-1]["end_time"]))
            checks["coverage"] = interval_iou(span, (0.0, duration)) >= cfg.min_overlap_iou
            checks["boundary"] = (
                span[0] <= cfg.tolerance_s and span[1] >= duration - cfg.tolerance_s
            )
        else:
            checks["coverage"] = False
            checks["boundary"] = False

        checks["alignment_score"] = (
            float(record.get("alignment_score", 0.0)) >= cfg.min_alignment_score
        )

        if cfg.require_asr_agreement:
            checks["asr"] = (
                float(record.get("asr_confidence", 1.0)) >= cfg.min_asr_confidence
            )

        subtitle = self._subtitle_check(record, cfg)
        if subtitle is not None:
            checks["subtitle"] = subtitle

        for axis in cfg.attribute_check_axes:
            agree = self._attribute_check(record, axis)
            if agree is not None:
                checks[f"attr_{axis}"] = agree

        return checks

    def _hard_fail(self, checks: Dict[str, bool], cfg: ConsensusConfig) -> bool:
        hard_keys = ["coverage", "alignment_score"]
        if cfg.require_asr_agreement:
            hard_keys.append("asr")
        if cfg.require_subtitle_agreement:
            hard_keys.append("subtitle")
        if cfg.require_attribute_agreement:
            hard_keys.extend(f"attr_{a}" for a in cfg.attribute_check_axes)
        return not all(checks[k] for k in hard_keys if k in checks)

    def _subtitle_check(self, record: Dict, cfg: ConsensusConfig) -> Optional[bool]:
        ref = self._subtitle_text(record)
        if not ref:
            return None
        hyp = record.get("text_itn") or record.get("text", "")
        if not hyp:
            return None
        sim = self._similarity(hyp, ref)
        record.setdefault("extra", {})["subtitle_similarity"] = round(sim, 4)
        return sim >= cfg.subtitle_min_similarity

    def _subtitle_text(self, record: Dict) -> str:
        extra = record.get("extra", {})
        if extra.get("subtitle_text"):
            return extra["subtitle_text"]
        path = extra.get("subtitle_path")
        if not path or not Path(path).exists():
            return ""
        cues = self._cues(path)
        if not cues:
            return ""
        start, end = record.get("start"), record.get("end")
        if start is None or end is None:
            return " ".join(c.text for c in cues)
        window = [c for c in cues if c.end > float(start) and c.start < float(end)]
        return " ".join(c.text for c in window)

    def _cues(self, path: str):
        if path in self._cue_cache:
            return self._cue_cache[path]
        if self._vtt_parser is None:
            from ....utils.subtitle import VTTParser

            self._vtt_parser = VTTParser()
        cues = self._vtt_parser.parse(path)
        self._cue_cache[path] = cues
        return cues

    def _attribute_check(self, record: Dict, axis: str) -> Optional[bool]:
        weak = (record.get("extra", {}).get("weak_labels") or {}).get(axis)
        model = record.get(axis)
        if not weak or not model:
            return None
        return weak == model

    def _similarity(self, a: str, b: str) -> float:
        na, nb = self._normalize(a), self._normalize(b)
        if not na or not nb:
            return 0.0
        return SequenceMatcher(None, na, nb).ratio()

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)).strip()
