import torch
import torch.nn as nn

from .base import BaseHead


class SpeakerHead(BaseHead):
    def __init__(
        self,
        input_dim: int,
        max_speakers: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__(name="speaker")
        self._max_speakers = max_speakers

        self._classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, max_speakers),
        )

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        return self._classifier(encoder_output)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn(predictions, targets)
