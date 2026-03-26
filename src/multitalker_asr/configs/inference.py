from dataclasses import dataclass, field
from typing import List, Optional

from .base import BaseConfig


@dataclass
class InferenceConfig(BaseConfig):
    audio_file: Optional[str] = None
    output_path: str = "data/output.json"
    att_context_size: List[int] = field(default_factory=lambda: [70, 13])
    batch_size: int = 1
    streaming_mode: bool = True
    parallel_speaker_strategy: bool = True
    discarded_frames: int = 8
    real_time_mode: bool = False
    manifest_file: Optional[str] = None
    deploy_mode: bool = False
    fix_prev_words_count: int = 10
    max_overlap_threshold: float = 0.25
    min_overlap_threshold: float = 0.01
    word_level_label: bool = True
    asr_confidence_threshold: float = 0.5
    streaming_batch_size: int = 1
    update_prev_words_sentence: int = 2
    max_no_spk_frames: int = 100
    max_concurrent_spks: int = 4
    vad_onset: float = 0.5
    vad_offset: float = 0.5
    use_sentence_render: bool = True
    ignored_initial_frame_steps: int = 0
    binary_diar_preds: bool = False
    max_num_of_spks: int = 4
    generate_realtime_scripts: bool = False
    print_path: str = "transcription.sh"
    word_window: int = 100
    verbose: bool = False
