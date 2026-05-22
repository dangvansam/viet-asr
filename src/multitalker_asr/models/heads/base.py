from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseHead(ABC, nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        pass
