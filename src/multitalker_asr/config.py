from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import OmegaConf


@dataclass
class ModelConfig:
    asr_model_path: str = "models/multitalker-parakeet-streaming-0.6b-v1.nemo"
    diar_model_path: str = "models/diar_streaming_sortformer_4spk-v2.1.nemo"
    diar_pretrained_name: str = "nvidia/diar_streaming_sortformer_4spk-v2.1"
    device: str = "cpu"
    cuda_id: int = -1
    use_amp: bool = True


@dataclass
class InferenceConfig:
    audio_file: Optional[str] = None
    output_path: str = "data/output.json"
    att_context_size: List[int] = field(default_factory=lambda: [70, 13])
    batch_size: int = 1
    streaming_mode: bool = True
    parallel_speaker_strategy: bool = True
    discarded_frames: int = 8
    real_time_mode: bool = False


@dataclass
class TrainingConfig:
    train_manifest: str = "data/train.json"
    val_manifest: str = "data/val.json"
    max_steps: int = 1000
    learning_rate: float = 1e-5
    weight_decay: float = 1e-3
    batch_size: int = 4
    accumulate_grad_batches: int = 4
    precision: int = 32  # 16 for GPU
    val_check_interval: int = 100


@dataclass
class DataConfig:
    input_csv: Optional[str] = None
    audio_dir: Optional[str] = None
    output_manifest: Optional[str] = None
    num_samples: int = 1000
    max_speakers: int = 2
    sample_rate: int = 16000


def get_config(config_type="inference", **kwargs):
    """Utility to get an OmegaConf object from dataclasses."""
    if config_type == "inference":
        base_cfg = OmegaConf.structured(InferenceConfig())
    elif config_type == "training":
        base_cfg = OmegaConf.structured(TrainingConfig())
    elif config_type == "data":
        base_cfg = OmegaConf.structured(DataConfig())
    else:
        raise ValueError(f"Unknown config type: {config_type}")

    # Merge overrides
    overrides = OmegaConf.create(kwargs)
    return OmegaConf.merge(base_cfg, overrides)
