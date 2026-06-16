"""
DataPipeline: orchestrates stage execution in config-defined order.
"""

from pathlib import Path
from typing import Dict, List, Optional, Type

from loguru import logger

from .base_stage import BaseStage
from .checkpoint import PipelineCheckpoint
from .config import PipelineConfig

# Lazy imports at class definition time to avoid import errors
# when not all stage dependencies are installed
from .stages.extract_audio import ExtractAudioStage
from .stages.audio_quality import AudioQualityStage
from .stages.vad import VADStage
from .stages.vad_diarize import VADDiarizeStage
from .stages.transcribe import TranscribeStage
from .stages.align import AlignStage
from .stages.consensus import ConsensusStage
from .stages.crawl_seed import CrawlSeedStage
from .stages.speaker_verify import SpeakerVerifyStage
from .stages.multitalker_reprocess import MultitalkerReprocessStage
from .stages.filter import FilterStage
from .stages.gender_classify import GenderClassifyStage
from .stages.enrich_labels import EnrichLabelsStage
from .stages.write_manifest import WriteManifestStage

STAGE_REGISTRY: Dict[str, Type[BaseStage]] = {
    "extract_audio": ExtractAudioStage,
    "audio_quality": AudioQualityStage,
    "vad": VADStage,
    "vad_diarize": VADDiarizeStage,
    "transcribe": TranscribeStage,
    "align": AlignStage,
    "consensus": ConsensusStage,
    "crawl_seed": CrawlSeedStage,
    "speaker_verify": SpeakerVerifyStage,
    "multitalker_reprocess": MultitalkerReprocessStage,
    "filter": FilterStage,
    "gender_classify": GenderClassifyStage,
    "enrich_labels": EnrichLabelsStage,
    "write_manifest": WriteManifestStage,
}

# First stages that generate records from scratch (no input file discovery).
GENERATOR_STAGES = {"enrich_labels", "crawl_seed"}

# Stages taking a sub-pipeline config; built lazily from PipelineConfig.stage_configs.
# spec: stage name → (config module, config class, stage module, stage class)
CONFIG_STAGE_SPECS: Dict[str, tuple] = {
    "multi_transcribe": (
        "...configs.transcript_pipeline", "TranscriptPipelineConfig",
        ".stages.multi_transcribe", "MultiBackendTranscribeStage",
    ),
    "normalize_text": (
        "...configs.itn_pipeline", "ITNPipelineConfig",
        ".stages.normalize_text", "NormalizeTextStage",
    ),
    "tag_attributes": (
        "...configs.attribute_pipeline", "AttributePipelineConfig",
        ".stages.tag_attributes", "TagAttributesStage",
    ),
    "transcript_refine": (
        "...configs.transcript_pipeline", "TranscriptPipelineConfig",
        ".stages.transcript_refine", "TranscriptRefineStage",
    ),
    "restore_pnc": (
        "...configs.llm", "LLMConfig",
        ".stages.restore_pnc", "RestorePnCStage",
    ),
}
CONFIG_STAGES = set(CONFIG_STAGE_SPECS)

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".mp4", ".mkv", ".webm", ".flac", ".m4a"}


class DataPipeline:
    """Orchestrate multi-stage data processing pipeline."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        checkpoint_dir = config.checkpoint_dir or str(
            Path(config.output_dir) / "checkpoints"
        )
        self._checkpoint = PipelineCheckpoint(checkpoint_dir)

        valid = set(STAGE_REGISTRY) | CONFIG_STAGES
        invalid = [n for n in config.stages if n not in valid]
        if invalid:
            raise ValueError(
                f"Unknown stage(s): {invalid}. Valid: {sorted(valid)}"
            )
        self._stages: List[BaseStage] = [
            self._construct_stage(name) for name in config.stages
        ]

    def _construct_stage(self, name: str) -> BaseStage:
        if name in CONFIG_STAGE_SPECS:
            return self._build_config_stage(name)
        return STAGE_REGISTRY[name]()

    def _build_config_stage(self, name: str) -> BaseStage:
        import importlib

        from omegaconf import OmegaConf

        cfg_mod, cfg_cls, stage_mod, stage_cls = CONFIG_STAGE_SPECS[name]
        config_module = importlib.import_module(cfg_mod, package=__package__)
        stage_module = importlib.import_module(stage_mod, package=__package__)
        ConfigClass = getattr(config_module, cfg_cls)
        StageClass = getattr(stage_module, stage_cls)

        raw = (self._config.stage_configs or {}).get(name, {})
        schema = OmegaConf.structured(ConfigClass())
        merged = OmegaConf.merge(schema, OmegaConf.create(raw))
        cfg = OmegaConf.to_object(merged)
        return StageClass(cfg)

    def run(self, input_files: Optional[List[str]] = None) -> int:
        """
        Run all stages in order.
        For raw pipeline: input_files is list of audio/video paths.
        For pretranscribed pipeline: input_files is [] (EnrichLabelsStage creates records).
        Returns total manifest entry count.
        """
        if input_files is not None:
            files = input_files
        elif self._config.stages and self._config.stages[0] in GENERATOR_STAGES:
            files = []
        else:
            files = self._discover_input_files()

        records = self._make_initial_records(files)
        logger.info(
            f"Pipeline '{self._config.pipeline_name}': {len(self._stages)} stages, "
            f"{len(records)} initial records"
        )

        for stage in self._stages:
            logger.info(f"Running stage: {stage.name} on {len(records)} records")
            try:
                records = stage.run(records, self._config, self._checkpoint)
            except Exception as e:
                logger.error(f"Stage {stage.name} failed: {e}")
                self._checkpoint.save_state()
                raise
            logger.success(f"Stage {stage.name} complete: {len(records)} records")

        return len(records)

    def run_records(self, records: List[Dict]) -> List[Dict]:
        """Run the constructed stages on pre-built records (no file discovery).

        Used by the streaming runner: records are injected per chunk and the
        stage list excludes record-generating / manifest-writing stages.
        """
        for stage in self._stages:
            records = stage.run(records, self._config, self._checkpoint)
        return records

    def _discover_input_files(self) -> List[str]:
        """
        Glob audio/video files from config.input_dir.
        Applies config.max_files limit if set.
        Returns sorted list of absolute paths.
        """
        input_dir = Path(self._config.input_dir)
        if not input_dir.exists():
            logger.warning(f"No input files found in {self._config.input_dir}")
            return []

        files: List[str] = []
        for ext in _AUDIO_EXTENSIONS:
            files.extend(str(p) for p in input_dir.rglob(f"*{ext}"))

        files = sorted(files)

        if not files:
            logger.warning(f"No input files found in {self._config.input_dir}")

        if self._config.max_files is not None:
            files = files[: self._config.max_files]

        return files

    def _make_initial_records(self, files: List[str]) -> List[Dict]:
        """Wrap file paths in minimal record dicts for the raw pipeline."""
        return [
            {
                "id": Path(f).stem,
                "audio_filepath": f,
                "source_video": f,
            }
            for f in files
        ]

    @classmethod
    def from_yaml(cls, yaml_path: str, **overrides) -> "DataPipeline":
        """
        Load config from YAML, apply kwargs overrides, return DataPipeline.
        Overrides: input_dir, output_dir, max_files, device
        """
        config = PipelineConfig.from_yaml(yaml_path)
        for k, v in overrides.items():
            if v is not None:
                setattr(config, k, v)
        return cls(config)
