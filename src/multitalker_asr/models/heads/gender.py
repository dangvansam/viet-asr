import torch
import torch.nn as nn

from .base import BaseHead


class GenderHead(BaseHead):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__(name="gender")
        self._num_classes = num_classes

        self._classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        pooled = encoder_output.mean(dim=1)
        return self._classifier(pooled)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(predictions, targets)
