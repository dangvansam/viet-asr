from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseInferenceEngine(ABC):
    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def infer(self, audio_path: str) -> List[Dict[str, Any]]:
        pass
