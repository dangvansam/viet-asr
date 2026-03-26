from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, eval_manifest: List[Dict[str, Any]]) -> Dict[str, Any]:
        pass


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


class BaseReporter(ABC):
    @abstractmethod
    def generate(
        self,
        results: Dict[str, Any],
        output_dir: str,
    ) -> None:
        pass
