from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class QualityResult:
    snr_db: float
    score: float = 0.0          # normalized [0, 1]
    ok: bool = True             # passes the min threshold (set by the stage)
    backend: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


class BaseQualityBackend(ABC):
    name: str = ""

    @abstractmethod
    def load(self, device: str = "cpu") -> None:
        ...

    @abstractmethod
    def score(self, audio: np.ndarray, sample_rate: int) -> QualityResult:
        ...

    def unload(self) -> None:
        return None

    @property
    def is_loaded(self) -> bool:
        return getattr(self, "_loaded", False)


class QualityBackendError(RuntimeError):
    pass
