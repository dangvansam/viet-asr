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
from .stages.vad_diarize import VADDiarizeStage
from .stages.transcribe import TranscribeStage
from .stages.align import AlignStage
from .stages.gender_classify import GenderClassifyStage
from .stages.enrich_labels import EnrichLabelsStage
from .stages.write_manifest import WriteManifestStage

STAGE_REGISTRY: Dict[str, Type[BaseStage]] = {
    "extract_audio": ExtractAudioStage,
    "vad_diarize": VADDiarizeStage,
    "transcribe": TranscribeStage,
    "align": AlignStage,
    "gender_classify": GenderClassifyStage,
    "enrich_labels": EnrichLabelsStage,
    "write_manifest": WriteManifestStage,
}

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".mp4", ".mkv", ".webm", ".flac", ".m4a"}


class DataPipeline:
    """Orchestrate multi-stage data processing pipeline."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        checkpoint_dir = config.checkpoint_dir or str(
            Path(config.output_dir) / "checkpoints"
        )
        self._checkpoint = PipelineCheckpoint(checkpoint_dir)

        invalid = [n for n in config.stages if n not in STAGE_REGISTRY]
        if invalid:
            raise ValueError(
                f"Unknown stage(s): {invalid}. Valid: {list(STAGE_REGISTRY)}"
            )
        self._stages: List[BaseStage] = [
            STAGE_REGISTRY[name]() for name in config.stages
        ]

    def run(self, input_files: Optional[List[str]] = None) -> int:
        """
        Run all stages in order.
        For raw pipeline: input_files is list of audio/video paths.
        For pretranscribed pipeline: input_files is [] (EnrichLabelsStage creates records).
        Returns total manifest entry count.
        """
        if input_files is not None:
            files = input_files
        elif self._config.stages and self._config.stages[0] == "enrich_labels":
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
