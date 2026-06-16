from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..prompt_embedding import PromptEmbedding
from .base import BaseConditioner, ConditionState


class PrependPromptCEConditioner(BaseConditioner):
    name = "prepend_prompt_ce"
    touches_encoder_input = True
    emits_decoder_tags = False
    produces_aux_logits = True

    def _build_layers(self) -> None:
        self._prompt_embed = PromptEmbedding(
            self._mt_cfg,
            vocabulary=self._registry.vocabulary,
            prefer_legacy_labels=True,
        )
        embed_dim = self._mt_cfg.prompt_embed_dim
        if embed_dim != self._encoder_input_dim:
            self._projection: Optional[nn.Linear] = nn.Linear(
                embed_dim, self._encoder_input_dim
            )
        else:
            self._projection = None
        self._classifier = nn.Linear(self._encoder_hidden_dim, self._registry.total_tokens)
        self._ce_loss = nn.CrossEntropyLoss()

    @property
    def num_positions(self) -> int:
        return self._prompt_embed.num_positions

    def apply_input(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        task_labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, ConditionState]:
        if task_labels is None:
            task_labels = self._zero_labels(features.shape[0], features.device)

        embeds = self._prompt_embed(task_labels)
        if self._projection is not None:
            embeds = self._projection(embeds)

        augmented = torch.cat([embeds, features], dim=1)
        augmented_lens = feature_lengths + self.num_positions
        state = ConditionState(num_prepended=self.num_positions)
        return augmented, augmented_lens, state

    def compute_aux_loss(
        self,
        encoder_out: torch.Tensor,
        task_labels: Optional[Dict[str, torch.Tensor]],
        state: ConditionState,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if state.num_prepended <= 0 or task_labels is None:
            device = encoder_out.device
            return torch.zeros((), device=device), {}

        prompt_out = encoder_out[:, : state.num_prepended, :]
        logits = self._classifier(prompt_out)

        total_loss = torch.zeros((), device=encoder_out.device)
        metrics: Dict[str, float] = {}
        batch_size = encoder_out.shape[0]

        for i, task in enumerate(self._mt_cfg.task_order):
            task_logits = logits[:, i, :]
            if task in task_labels:
                targets = task_labels[task]
            else:
                targets = torch.zeros(
                    batch_size, dtype=torch.long, device=encoder_out.device
                )
            start, _ = self._registry.get_task_range(task)
            shifted = targets + start
            loss_i = self._ce_loss(task_logits, shifted)
            total_loss = total_loss + loss_i
            with torch.no_grad():
                preds = task_logits.argmax(dim=-1)
                metrics[f"acc_{task}"] = (preds == shifted).float().mean().item()

        total_loss = total_loss / max(1, state.num_prepended)
        return total_loss, metrics

    def predict_attributes(
        self,
        encoder_out: torch.Tensor,
        state: ConditionState,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if state.num_prepended <= 0:
            return None
        prompt_out = encoder_out[:, : state.num_prepended, :]
        logits = self._classifier(prompt_out)
        predictions: Dict[str, torch.Tensor] = {}
        for i, task in enumerate(self._mt_cfg.task_order):
            task_logits = logits[:, i, :]
            start, end = self._registry.get_task_range(task)
            local = task_logits[:, start:end]
            predictions[task] = local.argmax(dim=-1)
        return predictions

    def _zero_labels(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        return {
            task: torch.zeros(batch_size, dtype=torch.long, device=device)
            for task in self._mt_cfg.task_order
        }
