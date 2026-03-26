from dataclasses import dataclass
from typing import Optional

from .base import BaseConfig


@dataclass
class EvalConfig(BaseConfig):
    diar_model_path: str = "models/diar_streaming_sortformer_4spk-v2.1.nemo"
    device: str = "cuda"
    cuda_id: int = 0
    source_manifest: Optional[str] = None
    eval_data_dir: str = "data/eval_diarization"
    num_samples: int = 200
    max_speakers: int = 4
    min_speakers: int = 2
    audio_dir: Optional[str] = None
    rttm_dir: Optional[str] = None
    streaming: bool = True
    batch_size: int = 1
    collar: float = 0.25
    ignore_overlap: bool = False
    eval_mode: str = "all"
    output_dir: str = "data/eval_results"
    generate_charts: bool = True
    generate_audacity_labels: bool = True
