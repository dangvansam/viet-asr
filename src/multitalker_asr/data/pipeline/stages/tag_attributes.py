from typing import Dict, List, Optional

from loguru import logger

from ....configs.attribute_pipeline import AttributeAxisConfig, AttributePipelineConfig
from ....utils.audio import AudioLoader
from ..attribute_backends import (
    AttributeEnsembler,
    AttributeResult,
    BaseAttributeBackend,
    build_attribute_backend,
)
from ..base_stage import BaseStage
from ..checkpoint import PipelineCheckpoint
from ..config import PipelineConfig


class TagAttributesStage(BaseStage):
    name = "tag_attributes"

    def __init__(
        self,
        config: AttributePipelineConfig,
        audio_loader: Optional[AudioLoader] = None,
        sample_rate: int = 16000,
    ):
        self._cfg = config
        self._sample_rate = sample_rate
        self._audio_loader = audio_loader or AudioLoader(target_sample_rate=sample_rate)
        self._ensemblers: Dict[str, AttributeEnsembler] = {}

    def _ensure_loaded(self) -> None:
        if self._ensemblers:
            return
        for axis_cfg in self._cfg.enabled_axes():
            ensembler = self._build_ensembler(axis_cfg)
            if ensembler is not None:
                self._ensemblers[axis_cfg.axis] = ensembler

    def _build_ensembler(
        self, axis_cfg: AttributeAxisConfig
    ) -> Optional[AttributeEnsembler]:
        backends: List[BaseAttributeBackend] = []
        for backend_cfg in axis_cfg.enabled_backends():
            try:
                backend = build_attribute_backend(
                    axis_cfg.axis, backend_cfg.name, **backend_cfg.kwargs
                )
                backend.load(device=backend_cfg.device)
                backends.append(backend)
            except Exception as exc:
                logger.warning(
                    f"Failed to load {axis_cfg.axis}/{backend_cfg.name}: {exc}"
                )
        if not backends:
            return None
        return AttributeEnsembler(backends, axis=axis_cfg.axis, strategy=axis_cfg.strategy)

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
                    metadata={"axes": list(self._ensemblers.keys())},
                )
                results.append(enriched)
            except Exception as exc:
                logger.error(f"TagAttributesStage failed on {record['id']}: {exc}")
                results.append(record)
        return results

    def _process_record(self, record: Dict) -> Dict:
        audio, sr = self._audio_loader.load(
            record["audio_filepath"],
            offset=record.get("offset", 0.0),
            duration=record.get("duration"),
        )
        text = record.get("text")
        enriched = dict(record)
        confidences = dict(enriched.get("attribute_confidence", {}))

        min_confidences = {
            cfg.axis: cfg.min_confidence for cfg in self._cfg.enabled_axes()
        }

        for axis, ensembler in self._ensemblers.items():
            result = ensembler.predict(audio, sr, text=text)
            if not result.label:
                continue
            if result.confidence < min_confidences.get(axis, 0.0):
                logger.debug(
                    f"Drop {axis}={result.label} conf={result.confidence:.3f} below threshold"
                )
                continue
            enriched[axis] = result.label
            confidences[axis] = result.confidence

        if confidences:
            enriched["attribute_confidence"] = confidences
        return enriched
