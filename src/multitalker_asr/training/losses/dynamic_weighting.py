from typing import Dict, Tuple

import torch
import torch.nn as nn
from loguru import logger


class DynamicLossWeighting(nn.Module):
    """Uncertainty-based dynamic loss weighting (Kendall et al. 2018).

    Learns log_var per task: loss = Σ (exp(-log_var_i) * loss_i + log_var_i)
    This auto-balances task magnitudes during multi-task training.
    """

    def __init__(self, task_names: list):
        super().__init__()
        self._task_names = task_names
        self.log_vars = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(1)) for name in task_names}
        )

    def forward(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute dynamically weighted total loss.

        Args:
            losses: {task_name: scalar loss tensor}

        Returns:
            (total_loss, stats) where stats has per-task weights and losses
        """
        total = torch.tensor(0.0, device=next(iter(losses.values())).device)
        stats = {}

        for name in self._task_names:
            if name not in losses:
                continue

            loss_i = losses[name]
            if torch.isnan(loss_i):
                logger.warning(f"Loss '{name}' is NaN, skipping")
                continue

            log_var = self.log_vars[name].clamp(-6.0, 6.0)
            precision = torch.exp(-log_var)
            weighted = precision * loss_i + log_var

            total = total + weighted
            stats[f"loss_{name}"] = loss_i.item()
            stats[f"weight_{name}"] = precision.item()
            stats[f"log_var_{name}"] = log_var.item()

        stats["loss_total"] = total.item()
        return total, stats
