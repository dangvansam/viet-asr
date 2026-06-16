from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .attribute_embedding import AttributeEmbedding
from .base import BaseConditioner, ConditionState


class FeatureConcatConditioner(BaseConditioner):
    """Nemotron-style: broadcast attribute vector over time, concat along feature dim."""

    name = "feature_concat"
    touches_encoder_input = True
    emits_decoder_tags = False
    produces_aux_logits = False

    def _build_layers(self) -> None:
        self._attr_embed = AttributeEmbedding(self._cfg, self._registry)
        projection_dim = self._cfg.feature_projection_dim or self._encoder_input_dim
        fused_dim = self._encoder_input_dim + self._attr_embed.total_dim
        self._projection = nn.Linear(fused_dim, projection_dim)

    def apply_input(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        task_labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, ConditionState]:
        batch_size = features.shape[0]
        time_steps = features.shape[1]
        device = features.device

        attr_vec = self._attr_embed(task_labels, batch_size=batch_size, device=device)
        if attr_vec.shape[-1] == 0:
            return features, feature_lengths, ConditionState()

        broadcast = self._attr_embed.broadcast_over_time(attr_vec, time_steps)
        fused = torch.cat([features, broadcast], dim=-1)
        projected = self._projection(fused)
        state = ConditionState(
            num_prepended=0,
            feature_concat_applied=True,
            extras={"attribute_dim": self._attr_embed.total_dim},
        )
        return projected, feature_lengths, state


class DecoderTagOnlyConditioner(BaseConditioner):
    """No encoder modification. Attribute information lives entirely in target tags."""

    name = "decoder_tag_only"
    touches_encoder_input = False
    emits_decoder_tags = True
    produces_aux_logits = False

    def _build_layers(self) -> None:
        return None

    def apply_input(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        task_labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, ConditionState]:
        return features, feature_lengths, ConditionState()


class HybridConditioner(BaseConditioner):
    """Feature-axis concat (input) + decoder tag tokens (output) + optional aux head."""

    name = "hybrid"
    touches_encoder_input = True
    emits_decoder_tags = True
    produces_aux_logits = True

    def _build_layers(self) -> None:
        self._attr_embed = AttributeEmbedding(self._cfg, self._registry)
        projection_dim = self._cfg.feature_projection_dim or self._encoder_input_dim
        fused_dim = self._encoder_input_dim + self._attr_embed.total_dim
        self._projection = nn.Linear(fused_dim, projection_dim)
        self._aux_classifier = nn.Linear(
            self._encoder_hidden_dim, self._registry.total_tokens
        )
        self._ce_loss = nn.CrossEntropyLoss()

    def apply_input(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        task_labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, ConditionState]:
        batch_size = features.shape[0]
        time_steps = features.shape[1]
        device = features.device

        attr_vec = self._attr_embed(task_labels, batch_size=batch_size, device=device)
        if attr_vec.shape[-1] == 0:
            return features, feature_lengths, ConditionState()

        broadcast = self._attr_embed.broadcast_over_time(attr_vec, time_steps)
        fused = torch.cat([features, broadcast], dim=-1)
        projected = self._projection(fused)
        state = ConditionState(
            num_prepended=0,
            feature_concat_applied=True,
            extras={"attribute_dim": self._attr_embed.total_dim},
        )
        return projected, feature_lengths, state

    def compute_aux_loss(
        self,
        encoder_out: torch.Tensor,
        task_labels: Optional[Dict[str, torch.Tensor]],
        state: ConditionState,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not self._cfg.enable_aux_head_loss or task_labels is None:
            return torch.zeros((), device=encoder_out.device), {}

        pooled = encoder_out.mean(dim=1)
        logits = self._aux_classifier(pooled)
        total_loss = torch.zeros((), device=encoder_out.device)
        metrics: Dict[str, float] = {}
        count = 0
        for task in self._mt_cfg.task_order:
            if task not in task_labels:
                continue
            start, end = self._registry.get_task_range(task)
            local = logits[:, start:end]
            targets = task_labels[task]
            loss_i = self._ce_loss(local, targets)
            total_loss = total_loss + loss_i
            with torch.no_grad():
                preds = local.argmax(dim=-1)
                metrics[f"acc_{task}"] = (preds == targets).float().mean().item()
            count += 1

        if count > 0:
            total_loss = total_loss / count
        return total_loss, metrics

    def predict_attributes(
        self,
        encoder_out: torch.Tensor,
        state: ConditionState,
    ) -> Optional[Dict[str, torch.Tensor]]:
        pooled = encoder_out.mean(dim=1)
        logits = self._aux_classifier(pooled)
        predictions: Dict[str, torch.Tensor] = {}
        for task in self._mt_cfg.task_order:
            start, end = self._registry.get_task_range(task)
            local = logits[:, start:end]
            predictions[task] = local.argmax(dim=-1)
        return predictions
