from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import BaseConfig
from .llm import LLMConfig


@dataclass
class ITNBackendConfig(BaseConfig):
    name: str = "funasr_itn"
    enabled: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("ITNBackendConfig.name must be non-empty")


@dataclass
class ITNPipelineConfig(BaseConfig):
    backends: List[ITNBackendConfig] = field(
        default_factory=lambda: [ITNBackendConfig(name="funasr_itn")]
    )
    strategy: str = "first_success"
    llm: Optional[LLMConfig] = None
    fallback_chain: bool = True

    def __post_init__(self):
        if not self.backends:
            raise ValueError("ITNPipelineConfig.backends must be non-empty")
        if self.strategy not in ("first_success", "vote", "llm_judge"):
            raise ValueError(
                f"Unknown strategy '{self.strategy}'. "
                f"Valid: first_success, vote, llm_judge"
            )
        if self.strategy == "llm_judge" and self.llm is None:
            raise ValueError("llm_judge strategy requires llm LLMConfig")

    def enabled_backends(self) -> List[ITNBackendConfig]:
        return [b for b in self.backends if b.enabled]
