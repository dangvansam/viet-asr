from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class BaseCollator(ABC):
    @abstractmethod
    def __call__(self, batch: List[Dict[str, Any]]) -> Tuple:
        pass
