from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class WordTiming:
    word: str
    start: float
    end: float
    confidence: Optional[float] = None


@dataclass
class ASRResult:
    text: str
    confidence: float = 1.0
    language: Optional[str] = None
    word_timings: Optional[List[WordTiming]] = None
    backend: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)
    vote_weight: Optional[float] = None


class BaseASRBackend(ABC):
    name: str = ""
    languages: List[str] = []

    @abstractmethod
    def load(self, device: str = "cpu") -> None:
        ...

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> ASRResult:
        ...

    def unload(self) -> None:
        return None

    @property
    def is_loaded(self) -> bool:
        return getattr(self, "_loaded", False)

    def supports_language(self, language: str) -> bool:
        if not self.languages:
            return True
        return language in self.languages


class ASRBackendError(RuntimeError):
    pass
