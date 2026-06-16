from dataclasses import dataclass, field
from typing import Dict, Optional

from .base import BaseConfig


@dataclass
class LLMConfig(BaseConfig):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 60.0
    max_retries: int = 3
    retry_backoff: float = 1.5
    extra_headers: Dict[str, str] = field(default_factory=dict)
    extra_params: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.provider:
            raise ValueError("LLMConfig.provider must be non-empty")
        if not self.model:
            raise ValueError("LLMConfig.model must be non-empty")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}")
        if self.timeout <= 0.0:
            raise ValueError(f"timeout must be > 0, got {self.timeout}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.retry_backoff < 1.0:
            raise ValueError(f"retry_backoff must be >= 1, got {self.retry_backoff}")

    @property
    def litellm_model(self) -> str:
        if "/" in self.model or self.provider in ("openai", "openai-compatible"):
            return self.model
        return f"{self.provider}/{self.model}"
