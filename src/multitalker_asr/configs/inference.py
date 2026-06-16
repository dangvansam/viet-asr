from dataclasses import dataclass, field
from typing import List, Optional

from .base import BaseConfig
from .streaming import ChunkPreset, StreamingProfile, preset_to_att_context


@dataclass
class InferenceConfig(BaseConfig):
    audio_file: Optional[str] = None
    output_path: str = "data/output.json"
    att_context_size: List[int] = field(default_factory=lambda: [56, 13])
    streaming_profile: Optional[StreamingProfile] = None
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

    def __post_init__(self):
        if self.streaming_profile is not None:
            profile_ctx = self.streaming_profile.att_context_size
            if self.att_context_size == [56, 13] and profile_ctx != self.att_context_size:
                self.att_context_size = profile_ctx

    @classmethod
    def with_preset(cls, preset: ChunkPreset, **kwargs) -> "InferenceConfig":
        profile = StreamingProfile(preset=preset)
        return cls(
            att_context_size=profile.att_context_size,
            streaming_profile=profile,
            **kwargs,
        )
