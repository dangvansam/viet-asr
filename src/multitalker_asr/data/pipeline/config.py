"""
Pipeline configuration dataclasses for the data processing pipeline.

Supports two pipeline modes:
- raw: long audio/video → extract_audio → vad_diarize → transcribe → align → gender_classify → write_manifest
- pretranscribed: pipe-delimited metadata → enrich_labels → transcribe → gender_classify → write_manifest
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from loguru import logger

try:
    from ..configs import BaseConfig
except ImportError:
    # Fallback if NeMo/package not installed in isolation
    @dataclass
    class BaseConfig:  # type: ignore[no-redef]
        pass


@dataclass
class VADConfig:
    min_duration: float = 3.0
    max_duration: float = 30.0
    model: str = "pyannote-onnx"
    hf_token: Optional[str] = None


@dataclass
class TranscribeConfig:
    model: str = "FunAudioLLM/Fun-ASR-MLT-Nano-2512"
    language: str = "auto"
    output_itn: bool = True
    itn_only: bool = False  # text-only normalization (pre-transcribed path)
    device: str = "cuda"
    batch_size: int = 8


@dataclass
class AlignConfig:
    model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    language: str = "Vietnamese"
    device: str = "cuda"
    dtype: str = "bfloat16"


@dataclass
class GenderConfig:
    url: str = "http://localhost:8000/predict"
    model: str = "ensemble"
    timeout: int = 30


@dataclass
class EnrichConfig:
    input_format: str = "pipe_delimited"  # "pipe_delimited" only for now
    metadata_path: str = ""  # path to pipe-delimited metadata file
    emotion_mapping: dict = field(default_factory=lambda: {
        "NEU": "neutral",
        "POS": "positive",
        "NEG": "negative",
    })


@dataclass
class ManifestConfig:
    output_filename: str = "manifest.jsonl"
    text_field: str = "text_itn"  # "text_itn" or "text"


@dataclass
class PipelineConfig(BaseConfig):
    pipeline_name: str = "raw"           # "raw" or "pretranscribed"
    stages: List[str] = field(default_factory=list)
    input_dir: str = ""
    output_dir: str = ""
    checkpoint_dir: str = ""             # defaults to output_dir/checkpoints if empty
    max_files: Optional[int] = None      # None = process all
    device: str = "cuda"
    vad: VADConfig = field(default_factory=VADConfig)
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    gender: GenderConfig = field(default_factory=GenderConfig)
    enrich: EnrichConfig = field(default_factory=EnrichConfig)
    manifest: ManifestConfig = field(default_factory=ManifestConfig)

    def __post_init__(self) -> None:
        # Default checkpoint_dir to output_dir/checkpoints when not set
        if not self.checkpoint_dir and self.output_dir:
            self.checkpoint_dir = str(Path(self.output_dir) / "checkpoints")

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load config from a YAML file, constructing nested dataclasses."""
        from omegaconf import OmegaConf

        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        raw = OmegaConf.load(yaml_path)
        # Convert to plain dict (struct=False so unknown keys are allowed)
        cfg_dict = OmegaConf.to_container(raw, resolve=True, throw_on_missing=False)

        # Warn about unknown top-level keys
        known_keys = {
            "pipeline_name", "stages", "input_dir", "output_dir",
            "checkpoint_dir", "max_files", "device",
            "vad", "transcribe", "align", "gender", "enrich", "manifest",
        }
        for key in cfg_dict:
            if key not in known_keys:
                logger.warning(f"Unknown config key: {key}")

        # Build nested sub-configs
        vad = VADConfig(**cfg_dict.pop("vad", {})) if "vad" in cfg_dict else VADConfig()
        transcribe = TranscribeConfig(**cfg_dict.pop("transcribe", {})) if "transcribe" in cfg_dict else TranscribeConfig()
        align = AlignConfig(**cfg_dict.pop("align", {})) if "align" in cfg_dict else AlignConfig()
        gender = GenderConfig(**cfg_dict.pop("gender", {})) if "gender" in cfg_dict else GenderConfig()
        enrich = EnrichConfig(**cfg_dict.pop("enrich", {})) if "enrich" in cfg_dict else EnrichConfig()
        manifest = ManifestConfig(**cfg_dict.pop("manifest", {})) if "manifest" in cfg_dict else ManifestConfig()

        return cls(
            vad=vad,
            transcribe=transcribe,
            align=align,
            gender=gender,
            enrich=enrich,
            manifest=manifest,
            **cfg_dict,
        )
