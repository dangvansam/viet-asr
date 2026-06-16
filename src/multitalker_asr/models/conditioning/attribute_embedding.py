from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ...configs.conditioning import ConditioningConfig
from ..prompt_embedding import TaskTokenRegistry


class AttributeEmbedding(nn.Module):
    """Feature-axis attribute embedding (Nemotron-style).

    For each axis with class count C and dim K, holds an Embedding(C, K).
    The forward produces a per-axis vector and stacks them along the feature axis,
    then broadcasts across time.
    """

    def __init__(
        self,
        config: ConditioningConfig,
        registry: TaskTokenRegistry,
    ):
        super().__init__()
        self._cfg = config
        self._registry = registry
        self._axes = [
            a for a in config.attribute_order if a in registry.task_names()
        ]
        self._embeds = nn.ModuleDict(
            {
                axis: nn.Embedding(
                    registry.get_class_count(axis),
                    config.attribute_dims[axis],
                )
                for axis in self._axes
            }
        )

    @property
    def axes(self) -> List[str]:
        return list(self._axes)

    @property
    def total_dim(self) -> int:
        return sum(self._cfg.attribute_dims[a] for a in self._axes)

    def forward(
        self,
        task_labels: Optional[Dict[str, torch.Tensor]],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        labels = task_labels or {}
        vectors: List[torch.Tensor] = []
        for axis in self._axes:
            if axis in labels:
                indices = labels[axis]
            else:
                indices = torch.zeros(batch_size, dtype=torch.long, device=device)
            vectors.append(self._embeds[axis](indices))
        if not vectors:
            return torch.zeros(batch_size, 0, device=device)
        return torch.cat(vectors, dim=-1)

    def broadcast_over_time(
        self,
        attribute_vector: torch.Tensor,
        time_steps: int,
    ) -> torch.Tensor:
        expanded = attribute_vector.unsqueeze(1).expand(-1, time_steps, -1)
        return expanded
