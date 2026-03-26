from dataclasses import dataclass
from typing import Optional

from .base import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    asr_model_path: str = "models/multitalker-parakeet-streaming-0.6b-v1.nemo"
    diar_model_path: str = "models/diar_streaming_sortformer_4spk-v2.1.nemo"
    diar_pretrained_name: str = "nvidia/diar_streaming_sortformer_4spk-v2.1"
    device: str = "cpu"
    cuda_id: int = -1
    use_amp: bool = True
    config_path: Optional[str] = None
    vocab_size: int = 2048
