from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class BaseASRModel(ABC):
    @abstractmethod
    def load(self, checkpoint_path: str) -> None:
        pass

    @abstractmethod
    def save(self, output_path: str) -> None:
        pass

    @abstractmethod
    def forward(self, audio: torch.Tensor) -> Dict[str, Any]:
        pass

    @abstractmethod
    def to(self, device: torch.device) -> "BaseASRModel":
        pass

    @abstractmethod
    def eval(self) -> "BaseASRModel":
        pass

    @abstractmethod
    def train(self) -> "BaseASRModel":
        pass
