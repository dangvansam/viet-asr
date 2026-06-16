from dataclasses import dataclass
from typing import Optional


@dataclass
class AuxLossScheduler:
    """Linear decay of auxiliary head loss weight over training epochs."""

    initial_weight: float = 1.0
    decay_epochs: int = 5
    min_weight: float = 0.0

    def __post_init__(self):
        if self.initial_weight < 0.0:
            raise ValueError(f"initial_weight must be >= 0, got {self.initial_weight}")
        if self.decay_epochs < 0:
            raise ValueError(f"decay_epochs must be >= 0, got {self.decay_epochs}")
        if self.min_weight < 0.0:
            raise ValueError(f"min_weight must be >= 0, got {self.min_weight}")
        if self.min_weight > self.initial_weight:
            raise ValueError(
                f"min_weight ({self.min_weight}) cannot exceed "
                f"initial_weight ({self.initial_weight})"
            )

    def weight(self, epoch: int) -> float:
        if epoch < 0:
            epoch = 0
        if self.decay_epochs == 0:
            return self.min_weight
        if epoch >= self.decay_epochs:
            return self.min_weight
        progress = epoch / self.decay_epochs
        return self.initial_weight - (self.initial_weight - self.min_weight) * progress

    def is_active(self, epoch: int, threshold: float = 1e-6) -> bool:
        return self.weight(epoch) > threshold
