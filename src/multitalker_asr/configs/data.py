from dataclasses import dataclass
from typing import Optional

from .base import BaseConfig


@dataclass
class DataConfig(BaseConfig):
    input_csv: Optional[str] = None
    audio_dir: Optional[str] = None
    output_manifest: Optional[str] = None
    num_samples: int = -1
    max_speakers: int = 4
    sample_rate: int = 16000
    use_on_the_fly_synthesis: bool = False
