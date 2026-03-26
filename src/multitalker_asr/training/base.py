from abc import ABC, abstractmethod
from typing import Optional


class BaseTrainer(ABC):
    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def save(self, output_path: Optional[str] = None) -> str:
        pass
