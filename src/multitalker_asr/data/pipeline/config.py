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
    backend: str = "silero"
    backend_kwargs: dict = field(default_factory=dict)
    device: str = "cpu"
    enable_prefilter: bool = True
    min_speech_ratio: float = 0.15
    enable_trim: bool = True
    trim_pad_s: float = 0.1
    min_gap_s: float = 0.2


@dataclass
class TranscribeConfig:
    model: str = "FunAudioLLM/Fun-ASR-MLT-Nano-2512"
    language: str = "auto"
    output_itn: bool = True
    itn_only: bool = False  # text-only normalization (pre-transcribed path)
    device: str = "cuda"
    batch_size: int = 8
    # Service mode: when set, the stage calls a remote OpenAI /v1/audio/transcriptions
    # server (the unified `openai_transcription` client) instead of loading the model
    # in-venv. The model's GPU is decided by that service. Leave empty for in-venv.
    base_url: str = ""
    sample_rate: int = 16000
    timeout: float = 120.0


@dataclass
class AlignConfig:
    model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    language: str = "Vietnamese"
    device: str = "cuda"
    dtype: str = "bfloat16"
    backend: str = "qwen3"
    backend_kwargs: dict = field(default_factory=dict)
    text_field: str = "text_itn"
    trim_to_words: bool = False
    trim_pad_s: float = 0.1


@dataclass
class ConsensusConfig:
    tolerance_s: float = 0.1
    min_overlap_iou: float = 0.5
    require_asr_agreement: bool = True
    min_asr_confidence: float = 0.0
    min_alignment_score: float = 0.8
    drop_on_disagreement: bool = True
    # Subtitle cross-check: ASR text vs crawl VTT subtitle similarity
    require_subtitle_agreement: bool = False
    subtitle_min_similarity: float = 0.5
    # Attribute cross-check: tag-derived weak label vs model-predicted label
    attribute_check_axes: List[str] = field(default_factory=list)
    require_attribute_agreement: bool = False


@dataclass
class AudioQualityConfig:
    enabled: bool = True
    backend: str = "snr"             # "snr" (numpy); dnsmos/deepfilternet can register later
    backend_kwargs: dict = field(default_factory=dict)
    device: str = "cpu"
    min_snr_db: float = 3.0          # drop files below this estimated SNR


@dataclass
class DiarizeConfig:
    backend: str = "pyannote"  # "pyannote" | "sortformer" | "vad_sv"
    model_path: str = "models/diar_streaming_sortformer_4spk-v2.1.nemo"  # sortformer .nemo
    device: str = "cuda"
    detect_overlap: bool = True
    overlap_min_s: float = 0.25  # min intersection to flag a segment as overlapping
    embed_url: str = "http://localhost:2010/embed"  # SV service for vad_sv labeling


@dataclass
class SpeakerVerifyConfig:
    enabled: bool = True
    url: str = "http://localhost:2010/embed"
    timeout: int = 30
    # Cosine merge threshold for clustering same-speaker utterances. The service's
    # 1:1 verify threshold (0.725) over-splits varied crawl clips; ~0.45 is a
    # calibrated default for grouping same-speaker segments. Tune per dataset.
    cluster_threshold: float = 0.45
    relabel_min: float = 0.45          # >= centroid cos → keep relabel; else maybe drop
    ambiguous_max: float = 0.30        # < and not overlap → drop (ambiguous/noise)
    min_cluster_size: int = 2
    drop_ambiguous: bool = True


@dataclass
class OverlapReprocessConfig:
    enabled: bool = False
    asr_model_path: str = "models/multitalker-vietnamese.nemo"
    diar_model_path: str = "models/diar_streaming_sortformer_4spk-v2.1.nemo"
    device: str = "cuda"
    min_overlap_speakers: int = 2


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
class CrawlSeedConfig:
    seed_path: str = ""  # path to seed.jsonl produced by scripts/ingest_crawl.py


@dataclass
class CrawlSourceConfig:
    """Stream source: the social-video-crawl DB (API or direct Postgres) + MinIO."""
    backend: str = "api"               # "api" (HTTP) | "db" (direct Postgres)
    api_url: str = ""                  # empty → CrawlAPIClient default / env
    api_token: str = ""                # empty → env CRAWL_API_TOKEN
    # Direct-Postgres creds (backend="db"); blank → crawl .env POSTGRES_* fallbacks.
    # NOTE: the DB port must be published on the host (compose `db.ports`) — the
    # container's internal POSTGRES_HOST=db is unreachable from the host pipeline.
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = ""
    db_password: str = ""
    db_name: str = ""
    platforms: List[str] = field(default_factory=list)  # empty → all platforms
    tag_family: List[str] = field(default_factory=list)  # ["stt","tts"] → tag_ids
    tag_ids: List[int] = field(default_factory=list)     # explicit override
    status: str = "completed"
    page_limit: int = 1000
    max_items: Optional[int] = None
    language: str = "vi"
    keep_meta: bool = True             # write meta/{task_id}.json for debugging
    # MinIO download creds (presigned keys with spaces need live-signing)
    minio_endpoint: str = ""
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = ""
    minio_region: str = ""
    minio_secure: bool = True


@dataclass
class StreamConfig:
    """Parallel on-the-fly streaming knobs (disk-bounded, GPU workers)."""
    batch_size: int = 32               # items per GPU-worker run() (amortizes model load)
    download_workers: int = 16         # prefetch threads
    gpu_workers: int = 3               # concurrent pipeline workers
    devices: List[str] = field(default_factory=lambda: ["cuda:1"])
    queue_size: int = 32               # bounded ready-queue → caps staged raw files on disk
    staging_dir: str = ""              # default {output_dir}/_staging
    keep_raw: bool = False             # keep staged source audio after processing
    keep_extracted: bool = False       # keep 16k extracted intermediates
    manifest_flush_every: int = 5      # rewrite manifest every N batches


@dataclass
class ManifestConfig:
    output_filename: str = "manifest.jsonl"
    text_field: str = "text_itn"  # "text_itn" or "text"
    shard_by: Optional[str] = None  # e.g. "platform" → one manifest per value + merged
    dataset_metadata_filename: str = "dataset_metadata.jsonl"
    dataset_summary_filename: str = "dataset_summary.json"


@dataclass
class PipelineConfig(BaseConfig):
    pipeline_name: str = "raw"           # "raw" or "pretranscribed"
    stages: List[str] = field(default_factory=list)
    input_dir: str = ""
    output_dir: str = ""
    checkpoint_dir: str = ""             # defaults to output_dir/checkpoints if empty
    max_files: Optional[int] = None      # None = process all
    device: str = "cuda"
    concurrency: int = 8                  # per-stage across-record client concurrency (HTTP I/O)
    vad: VADConfig = field(default_factory=VADConfig)
    audio_quality: AudioQualityConfig = field(default_factory=AudioQualityConfig)
    diarize: DiarizeConfig = field(default_factory=DiarizeConfig)
    speaker_verify: SpeakerVerifyConfig = field(default_factory=SpeakerVerifyConfig)
    overlap_reprocess: OverlapReprocessConfig = field(default_factory=OverlapReprocessConfig)
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    gender: GenderConfig = field(default_factory=GenderConfig)
    enrich: EnrichConfig = field(default_factory=EnrichConfig)
    crawl_seed: CrawlSeedConfig = field(default_factory=CrawlSeedConfig)
    crawl_source: CrawlSourceConfig = field(default_factory=CrawlSourceConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    manifest: ManifestConfig = field(default_factory=ManifestConfig)
    stage_configs: dict = field(default_factory=dict)  # raw sub-pipeline configs by stage name

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
            "vad", "audio_quality", "diarize", "speaker_verify", "overlap_reprocess",
            "transcribe", "align", "consensus",
            "gender", "enrich", "crawl_seed", "crawl_source", "stream",
            "manifest", "stage_configs",
        }
        for key in cfg_dict:
            if key not in known_keys:
                logger.warning(f"Unknown config key: {key}")

        # Build nested sub-configs
        vad = VADConfig(**cfg_dict.pop("vad", {})) if "vad" in cfg_dict else VADConfig()
        audio_quality = AudioQualityConfig(**cfg_dict.pop("audio_quality", {})) if "audio_quality" in cfg_dict else AudioQualityConfig()
        diarize = DiarizeConfig(**cfg_dict.pop("diarize", {})) if "diarize" in cfg_dict else DiarizeConfig()
        speaker_verify = SpeakerVerifyConfig(**cfg_dict.pop("speaker_verify", {})) if "speaker_verify" in cfg_dict else SpeakerVerifyConfig()
        overlap_reprocess = OverlapReprocessConfig(**cfg_dict.pop("overlap_reprocess", {})) if "overlap_reprocess" in cfg_dict else OverlapReprocessConfig()
        transcribe = TranscribeConfig(**cfg_dict.pop("transcribe", {})) if "transcribe" in cfg_dict else TranscribeConfig()
        align = AlignConfig(**cfg_dict.pop("align", {})) if "align" in cfg_dict else AlignConfig()
        consensus = ConsensusConfig(**cfg_dict.pop("consensus", {})) if "consensus" in cfg_dict else ConsensusConfig()
        gender = GenderConfig(**cfg_dict.pop("gender", {})) if "gender" in cfg_dict else GenderConfig()
        enrich = EnrichConfig(**cfg_dict.pop("enrich", {})) if "enrich" in cfg_dict else EnrichConfig()
        crawl_seed = CrawlSeedConfig(**cfg_dict.pop("crawl_seed", {})) if "crawl_seed" in cfg_dict else CrawlSeedConfig()
        crawl_source = CrawlSourceConfig(**cfg_dict.pop("crawl_source", {})) if "crawl_source" in cfg_dict else CrawlSourceConfig()
        stream = StreamConfig(**cfg_dict.pop("stream", {})) if "stream" in cfg_dict else StreamConfig()
        manifest = ManifestConfig(**cfg_dict.pop("manifest", {})) if "manifest" in cfg_dict else ManifestConfig()

        return cls(
            vad=vad,
            audio_quality=audio_quality,
            diarize=diarize,
            speaker_verify=speaker_verify,
            overlap_reprocess=overlap_reprocess,
            transcribe=transcribe,
            align=align,
            consensus=consensus,
            gender=gender,
            enrich=enrich,
            crawl_seed=crawl_seed,
            crawl_source=crawl_source,
            stream=stream,
            manifest=manifest,
            **cfg_dict,
        )
