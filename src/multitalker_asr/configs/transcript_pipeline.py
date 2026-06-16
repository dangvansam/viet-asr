from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import BaseConfig
from .llm import LLMConfig


@dataclass
class ASRBackendConfig(BaseConfig):
    name: str = "funasr"
    weight: float = 1.0
    device: str = "cpu"
    enabled: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("ASRBackendConfig.name must be non-empty")
        if self.weight < 0.0:
            raise ValueError(f"weight must be >= 0, got {self.weight}")


@dataclass
class TranscriptPipelineConfig(BaseConfig):
    backends: List[ASRBackendConfig] = field(
        default_factory=lambda: [ASRBackendConfig(name="funasr")]
    )
    ensemble_strategy: str = "single"
    min_agree: int = 2
    similarity_threshold: float = 0.85
    llm_judge: Optional[LLMConfig] = None
    fallback_text: str = ""
    # transcript_refine / rover: use the crawl VTT subtitle as a high-weight voter,
    # and treat the incoming record transcript (Fun-ASR) as the anchor.
    use_subtitle: bool = True
    subtitle_weight: float = 1.5
    include_primary: bool = True
    # Forced alignment: give word timestamps to backends that return text-only
    # (vietasr, qwen3) so they join word-level ROVER. "" / "none" disables.
    align_backend: str = "mms_fa"          # "mms_fa" (in-venv) | "qwen3_service" | "nemo_nfa"
    align_device: str = "cuda"
    align_language: str = "Vietnamese"
    align_kwargs: Dict[str, Any] = field(default_factory=dict)
    segment_concurrency: int = 8           # parallel segments per stage (HTTP I/O fan-out)
    asr_concurrency: int = 4               # parallel ASR/align calls within one segment

    def __post_init__(self):
        if not self.backends:
            raise ValueError("TranscriptPipelineConfig.backends must be non-empty")
        if self.ensemble_strategy not in ("single", "vote", "rover", "llm_judge"):
            raise ValueError(
                f"Unknown ensemble_strategy '{self.ensemble_strategy}'. "
                f"Valid: single, vote, rover, llm_judge"
            )
        if self.ensemble_strategy == "llm_judge" and self.llm_judge is None:
            raise ValueError("llm_judge strategy requires llm_judge LLMConfig")
        if self.min_agree < 1:
            raise ValueError(f"min_agree must be >= 1, got {self.min_agree}")
        if not 0.0 < self.similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in (0,1], got {self.similarity_threshold}"
            )

    def enabled_backends(self) -> List[ASRBackendConfig]:
        return [b for b in self.backends if b.enabled]
