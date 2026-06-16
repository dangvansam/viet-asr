from typing import Dict, List, Optional

from loguru import logger

from ....configs.itn_pipeline import ITNPipelineConfig
from ....utils.llm_client import LLMClient
from ..base_stage import BaseStage
from ..checkpoint import PipelineCheckpoint
from ..config import PipelineConfig
from ..itn_backends import BaseITNBackend, ITNStrategyFactory, build_itn_backend


class NormalizeTextStage(BaseStage):
    name = "normalize_text"

    def __init__(self, config: ITNPipelineConfig):
        self._cfg = config
        self._backends: List[BaseITNBackend] = []
        self._strategy = None

    def _ensure_loaded(self) -> None:
        if self._backends:
            return
        for backend_cfg in self._cfg.enabled_backends():
            backend = build_itn_backend(backend_cfg.name, **backend_cfg.kwargs)
            backend.load()
            self._backends.append(backend)
        llm_client = None
        if self._cfg.strategy == "llm_judge" and self._cfg.llm is not None:
            llm_client = LLMClient(self._cfg.llm)
        self._strategy = ITNStrategyFactory.build(
            self._cfg.strategy, llm_client=llm_client
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
        results: List[Dict] = list(done)
        for record in to_process:
            try:
                enriched = self._process_record(record)
                checkpoint.mark_processed(
                    record["id"],
                    self.name,
                    metadata={"backend": enriched.get("itn_backend")},
                )
                results.append(enriched)
            except Exception as exc:
                logger.error(f"NormalizeTextStage failed on {record['id']}: {exc}")
                results.append(record)
        return results

    def _process_record(self, record: Dict) -> Dict:
        text = record.get("text", "")
        language = record.get("language", "vi")
        if not text:
            return record
        itn_result = self._strategy.apply(self._backends, text, language)
        enriched = dict(record)
        enriched["text_itn"] = itn_result.text_itn
        enriched["itn_backend"] = itn_result.backend
        enriched["itn_confidence"] = itn_result.confidence
        return enriched
