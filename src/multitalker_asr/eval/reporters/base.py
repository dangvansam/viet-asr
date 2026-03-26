from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseReporter(ABC):
    @abstractmethod
    def generate(
        self,
        results: Dict[str, Any],
        output_dir: str,
    ) -> None:
        pass
