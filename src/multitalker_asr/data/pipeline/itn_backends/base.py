from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ITNResult:
    text_itn: str
    text_spoken: Optional[str] = None
    confidence: float = 1.0
    backend: str = ""
    language: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class BaseITNBackend(ABC):
    name: str = ""
    languages: List[str] = []

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def normalize(self, text: str, language: str = "vi") -> ITNResult:
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


class ITNBackendError(RuntimeError):
    pass
