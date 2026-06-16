from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .base import BaseConfig


class ConditioningStrategy(str, Enum):
    PREPEND_PROMPT_CE = "prepend_prompt_ce"
    DECODER_TAG_ONLY = "decoder_tag_only"
    FEATURE_CONCAT = "feature_concat"
    HYBRID = "hybrid"


class TagEmissionMode(str, Enum):
    NEVER = "never"
    ALWAYS = "always"
    AUTO = "auto"


@dataclass
class ConditioningConfig(BaseConfig):
    strategy: ConditioningStrategy = ConditioningStrategy.HYBRID

    attribute_order: List[str] = field(
        default_factory=lambda: ["language", "emotion", "gender", "age", "region"]
    )

    attribute_dims: Dict[str, int] = field(
        default_factory=lambda: {
            "language": 16,
            "emotion": 16,
            "gender": 8,
            "age": 12,
            "region": 12,
        }
    )

    p_condition: float = 0.5
    p_drop_attribute: float = 0.1

    enable_aux_head_loss: bool = True
    aux_head_loss_weight: float = 1.0
    aux_head_loss_decay_epochs: int = 5

    tag_emission_mode: TagEmissionMode = TagEmissionMode.AUTO
    tag_position: str = "after_terminal_punct"
    strip_attribute_tags: bool = False

    feature_projection_dim: Optional[int] = None
    broadcast_temporal: bool = True

    def __post_init__(self):
        if isinstance(self.strategy, str):
            self.strategy = ConditioningStrategy(self.strategy)
        if isinstance(self.tag_emission_mode, str):
            self.tag_emission_mode = TagEmissionMode(self.tag_emission_mode)

        if not 0.0 <= self.p_condition <= 1.0:
            raise ValueError(f"p_condition must be in [0,1], got {self.p_condition}")
        if not 0.0 <= self.p_drop_attribute <= 1.0:
            raise ValueError(
                f"p_drop_attribute must be in [0,1], got {self.p_drop_attribute}"
            )
        if self.aux_head_loss_weight < 0.0:
            raise ValueError(
                f"aux_head_loss_weight must be >= 0, got {self.aux_head_loss_weight}"
            )
        if self.aux_head_loss_decay_epochs < 0:
            raise ValueError(
                f"aux_head_loss_decay_epochs must be >= 0, "
                f"got {self.aux_head_loss_decay_epochs}"
            )
        if self.tag_position not in ("after_terminal_punct", "prefix", "suffix"):
            raise ValueError(
                f"tag_position must be one of 'after_terminal_punct', 'prefix', 'suffix', "
                f"got '{self.tag_position}'"
            )

        for attr in self.attribute_order:
            if attr not in self.attribute_dims:
                raise ValueError(
                    f"attribute '{attr}' in attribute_order but missing from attribute_dims"
                )
            if self.attribute_dims[attr] <= 0:
                raise ValueError(
                    f"attribute_dims['{attr}'] must be > 0, got {self.attribute_dims[attr]}"
                )

    @property
    def emits_decoder_tags(self) -> bool:
        return self.strategy in (
            ConditioningStrategy.DECODER_TAG_ONLY,
            ConditioningStrategy.HYBRID,
        )

    @property
    def uses_feature_concat(self) -> bool:
        return self.strategy in (
            ConditioningStrategy.FEATURE_CONCAT,
            ConditioningStrategy.HYBRID,
        )

    @property
    def uses_prepend_prompt(self) -> bool:
        return self.strategy == ConditioningStrategy.PREPEND_PROMPT_CE

    @property
    def total_attribute_dim(self) -> int:
        return sum(self.attribute_dims[a] for a in self.attribute_order)


class ConditionerFactory:
    _builders: Dict[ConditioningStrategy, "callable"] = {}

    @classmethod
    def register(cls, strategy: ConditioningStrategy, builder=None):
        if builder is not None:
            cls._builders[strategy] = builder
            return builder

        def _decorator(fn):
            cls._builders[strategy] = fn
            return fn

        return _decorator

    @classmethod
    def build(cls, cfg: ConditioningConfig, **kwargs):
        if cfg.strategy not in cls._builders:
            raise NotImplementedError(
                f"No conditioner registered for strategy '{cfg.strategy.value}'. "
                f"Registered: {[s.value for s in cls._builders.keys()]}"
            )
        return cls._builders[cfg.strategy](cfg, **kwargs)

    @classmethod
    def registered_strategies(cls) -> List[ConditioningStrategy]:
        return list(cls._builders.keys())

    @classmethod
    def unregister(cls, strategy: ConditioningStrategy) -> None:
        cls._builders.pop(strategy, None)
