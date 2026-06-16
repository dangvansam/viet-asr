from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AlignedWord:
    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0


@dataclass
class AlignResult:
    words: List[AlignedWord]
    score: float = 0.0
    backend: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def span(self):
        if not self.words:
            return None
        return self.words[0].start_time, self.words[-1].end_time

    def to_records(self) -> List[Dict[str, Any]]:
        return [
            {"text": w.text, "start_time": w.start_time, "end_time": w.end_time}
            for w in self.words
        ]


def alignment_score(words: List[AlignedWord]) -> float:
    """Fraction of words with valid timestamps (start >= 0 and end > start)."""
    if not words:
        return 0.0
    valid = sum(1 for w in words if w.start_time >= 0 and w.end_time > w.start_time)
    return valid / len(words)


class BaseAlignBackend(ABC):
    name: str = ""
    languages: List[str] = []

    @abstractmethod
    def load(self, device: str = "cpu") -> None:
        ...

    @abstractmethod
    def align(self, audio_path: str, text: str, language: str) -> AlignResult:
        ...

    def unload(self) -> None:
        return None

    @property
    def is_loaded(self) -> bool:
        return getattr(self, "_loaded", False)

    def supports_language(self, language: str) -> bool:
        if not self.languages:
            return True
        return language.lower() in {lang.lower() for lang in self.languages}


class AlignBackendError(RuntimeError):
    pass
