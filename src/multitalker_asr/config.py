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
    config_path: Optional[str] = None
    vocab_size: int = 2048


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


@dataclass
class TrainingConfig:
    train_manifest: str = "data/train.json"
    val_manifest: str = "data/val.json"
    max_steps: int = -1
    max_epochs: int = 100
    learning_rate: float = 1e-5
    weight_decay: float = 1e-3
    batch_size: int = 8
    accumulate_grad_batches: int = 2
    precision: int = 32
    val_check_interval: Optional[int] = None
    output_path: Optional[str] = None
    tokenizer_dir: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    # Synthesis parameters
    use_on_the_fly_synthesis: bool = False
    max_speakers: int = 2
    synthesis_num_workers: int = 4


@dataclass
class DataConfig:
    input_csv: Optional[str] = None
    audio_dir: Optional[str] = None
    output_manifest: Optional[str] = None
    num_samples: int = -1
    max_speakers: int = 4
    sample_rate: int = 16000
    use_on_the_fly_synthesis: bool = False


@dataclass
class EvalConfig:
    # Model
    diar_model_path: str = "models/diar_streaming_sortformer_4spk-v2.1.nemo"
    device: str = "cuda"
    cuda_id: int = 0

    # Data synthesis
    source_manifest: Optional[str] = None
    eval_data_dir: str = "data/eval_diarization"
    num_samples: int = 200
    max_speakers: int = 4
    min_speakers: int = 2

    # Pre-existing eval data (skip synthesis)
    audio_dir: Optional[str] = None
    rttm_dir: Optional[str] = None

    # Inference
    streaming: bool = True
    batch_size: int = 1

    # Evaluation
    collar: float = 0.25
    ignore_overlap: bool = False
    eval_mode: str = "all"  # "full", "fair", "forgiving", "all"

    # Output
    output_dir: str = "data/eval_results"
    generate_charts: bool = True
    generate_audacity_labels: bool = True


def get_config(config_type="inference", **kwargs):
    """Utility to get an OmegaConf object from dataclasses."""
    if config_type == "inference":
        base_cfg = OmegaConf.structured(InferenceConfig())
    elif config_type == "training":
        base_cfg = OmegaConf.structured(TrainingConfig())
    elif config_type == "data":
        base_cfg = OmegaConf.structured(DataConfig())
    elif config_type == "eval":
        base_cfg = OmegaConf.structured(EvalConfig())
    else:
        raise ValueError(f"Unknown config type: {config_type}")

    # Merge overrides
    overrides = OmegaConf.create(kwargs)
    return OmegaConf.merge(base_cfg, overrides)
