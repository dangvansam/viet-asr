from typing import Dict, Tuple

import torch
import torch.nn as nn
from loguru import logger


class MultiTaskLoss(nn.Module):
    """Combines RNNT loss + prompt CE loss with configurable weights.

    Loss = rnnt_weight * loss_rnnt + prompt_weight * loss_prompt
    """

    def __init__(self, rnnt_weight: float = 1.0, prompt_weight: float = 1.0):
        super().__init__()
        if rnnt_weight < 0 or prompt_weight < 0:
            raise ValueError(
                f"Loss weights must be >= 0, got rnnt={rnnt_weight}, prompt={prompt_weight}"
            )
        self.rnnt_weight = rnnt_weight
        self.prompt_weight = prompt_weight

    def forward(
        self,
        loss_rnnt: torch.Tensor,
        loss_prompt: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weighted total loss.

        Returns:
            (total_loss, stats_dict) where stats_dict has individual losses for logging.
        """
        # Guard against NaN
        if torch.isnan(loss_rnnt):
            logger.warning("RNNT loss is NaN, replacing with 0")
            loss_rnnt = torch.tensor(0.0, device=loss_prompt.device)
        if torch.isnan(loss_prompt):
            logger.warning("Prompt CE loss is NaN, replacing with 0")
            loss_prompt = torch.tensor(0.0, device=loss_rnnt.device)

        total = self.rnnt_weight * loss_rnnt + self.prompt_weight * loss_prompt

        stats = {
            "loss_total": total.item(),
            "loss_rnnt": loss_rnnt.item(),
            "loss_prompt": loss_prompt.item(),
            "weight_rnnt": self.rnnt_weight,
            "weight_prompt": self.prompt_weight,
        }
        return total, stats
