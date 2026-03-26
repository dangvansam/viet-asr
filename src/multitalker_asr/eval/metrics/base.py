from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseMetric(ABC):
    @abstractmethod
    def compute(
        self,
        reference: Any,
        hypothesis: Any,
    ) -> Dict[str, float]:
        pass

    @abstractmethod
    def aggregate(
        self,
        per_file_results: List[Dict[str, float]],
    ) -> Dict[str, float]:
        pass
