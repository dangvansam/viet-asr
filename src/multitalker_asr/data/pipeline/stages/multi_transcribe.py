from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from ....configs.transcript_pipeline import TranscriptPipelineConfig
from ....utils.audio import AudioLoader
from ....utils.llm_client import LLMClient
from ..asr_backends import (
    ASREnsemblerFactory,
    ASRResult,
    BaseASRBackend,
    build_asr_backend,
)
from ..base_stage import BaseStage
from ..checkpoint import PipelineCheckpoint
from ..config import PipelineConfig
from ..parallel import parallel_map


class MultiBackendTranscribeStage(BaseStage):
    name = "multi_transcribe"

    def __init__(
        self,
        config: TranscriptPipelineConfig,
        audio_loader: Optional[AudioLoader] = None,
        sample_rate: int = 16000,
    ):
        self._cfg = config
        self._sample_rate = sample_rate
        self._audio_loader = audio_loader or AudioLoader(target_sample_rate=sample_rate)
        self._backends: List[BaseASRBackend] = []
        self._ensembler = None

    def _ensure_loaded(self) -> None:
        if self._backends:
            return
        for backend_cfg in self._cfg.enabled_backends():
            backend = build_asr_backend(backend_cfg.name, **backend_cfg.kwargs)
            backend.load(device=backend_cfg.device)
            self._backends.append(backend)
        llm_client = None
        if self._cfg.ensemble_strategy == "llm_judge" and self._cfg.llm_judge is not None:
            llm_client = LLMClient(self._cfg.llm_judge)
        self._ensembler = ASREnsemblerFactory.build(
            strategy=self._cfg.ensemble_strategy,
            llm_client=llm_client,
            min_agree=self._cfg.min_agree,
            similarity_threshold=self._cfg.similarity_threshold,
        )

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        to_process, done = self._skip_processed(records, checkpoint)
        if not to_process:
            return done

        self._ensure_loaded()

        def safe(record: Dict) -> Optional[Dict]:
            try:
                return self._process_record(record)
            except Exception as exc:
                logger.error(f"MultiBackendTranscribeStage failed on {record['id']}: {exc}")
                return None

        workers = getattr(config, "concurrency", 1) or 1
        enriched_list = parallel_map(safe, to_process, workers)
        results: List[Dict] = list(done)
        for record, enriched in zip(to_process, enriched_list):
            if enriched is None:
                continue
            checkpoint.mark_processed(
                record["id"],
                self.name,
                metadata={
                    "backends": [b.backend for b in enriched.get("_hypotheses", [])],
                    "ensemble": self._cfg.ensemble_strategy,
                },
            )
            results.append(enriched)
        return results

    def _process_record(self, record: Dict) -> Dict:
        audio, sr = self._audio_loader.load(
            record["audio_filepath"],
            offset=record.get("offset", 0.0),
            duration=record.get("duration"),
        )
        language = record.get("language")
        hypotheses = self._run_backends(audio, sr, language)
        final = self._ensembler.combine(hypotheses)
        enriched = dict(record)
        enriched["text"] = final.text
        enriched["asr_confidence"] = final.confidence
        if final.language and not enriched.get("language"):
            enriched["language"] = final.language
        enriched["_asr_backends"] = [
            {"backend": h.backend, "text": h.text, "confidence": h.confidence}
            for h in hypotheses
        ]
        raw_candidates = final.raw.get("candidates") if isinstance(final.raw, dict) else None
        candidates = [c for c in (raw_candidates or []) if isinstance(c, str) and c.strip()]
        if len(candidates) > 1:
            extra = dict(enriched.get("extra") or {})
            extra["asr_candidates_nbest"] = candidates
            enriched["extra"] = extra
        return enriched

    def _run_backends(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str],
    ) -> List[ASRResult]:
        results: List[ASRResult] = []
        for backend in self._backends:
            try:
                results.append(backend.transcribe(audio, sample_rate, language=language))
            except Exception as exc:
                logger.warning(f"ASR backend '{backend.name}' failed: {exc}")
        return results
