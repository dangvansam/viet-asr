from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...configs.conditioning import ConditioningConfig
from ...configs.multitask import MultiTaskConfig
from ..prompt_embedding import TaskTokenRegistry


@dataclass
class ConditionState:
    num_prepended: int = 0
    feature_concat_applied: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)


class BaseConditioner(nn.Module, ABC):
    """Pluggable attribute conditioning module.

    Strategies vary in how attribute information enters the network:
        * input side — prepend prompt embeddings, broadcast & concat along feature dim
        * output side — emit tag tokens via decoder vocabulary, optional auxiliary heads

    Each concrete subclass declares whether it touches the encoder input, emits
    decoder tag tokens, and produces auxiliary classification logits.
    """

    name: str = ""
    touches_encoder_input: bool = False
    emits_decoder_tags: bool = False
    produces_aux_logits: bool = False

    def __init__(
        self,
        config: ConditioningConfig,
        multitask_config: MultiTaskConfig,
        registry: TaskTokenRegistry,
    ):
        super().__init__()
        self._cfg = config
        self._mt_cfg = multitask_config
        self._registry = registry
        self._encoder_hidden_dim: Optional[int] = None
        self._encoder_input_dim: Optional[int] = None

    @property
    def config(self) -> ConditioningConfig:
        return self._cfg

    @property
    def registry(self) -> TaskTokenRegistry:
        return self._registry

    def initialize_layers(self, encoder_input_dim: int, encoder_hidden_dim: int) -> None:
        self._encoder_input_dim = encoder_input_dim
        self._encoder_hidden_dim = encoder_hidden_dim
        self._build_layers()

    @abstractmethod
    def _build_layers(self) -> None:
        ...

    @abstractmethod
    def apply_input(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        task_labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, ConditionState]:
        ...

    def strip_speech(
        self,
        encoder_out: torch.Tensor,
        encoder_lengths: torch.Tensor,
        state: ConditionState,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.num_prepended <= 0:
            return encoder_out, encoder_lengths
        return (
            encoder_out[:, state.num_prepended :, :],
            encoder_lengths - state.num_prepended,
        )

    def compute_aux_loss(
        self,
        encoder_out: torch.Tensor,
        task_labels: Optional[Dict[str, torch.Tensor]],
        state: ConditionState,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = encoder_out.device
        return torch.zeros((), device=device), {}

    def predict_attributes(
        self,
        encoder_out: torch.Tensor,
        state: ConditionState,
    ) -> Optional[Dict[str, torch.Tensor]]:
        return None

    def decorate_target(
        self,
        text: str,
        attribute_labels: Dict[str, int],
    ) -> str:
        if not self.emits_decoder_tags:
            return text
        return self._registry.append_tags(text, attribute_labels)

    def parse_target_tags(self, text: str) -> Tuple[str, Dict[str, int]]:
        if not self.emits_decoder_tags:
            return text, {}
        return self._registry.parse_tags_from_text(text)
