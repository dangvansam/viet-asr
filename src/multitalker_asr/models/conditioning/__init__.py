from ...configs.conditioning import ConditionerFactory, ConditioningStrategy
from .attribute_embedding import AttributeEmbedding
from .aux_loss_scheduler import AuxLossScheduler
from .base import BaseConditioner, ConditionState
from .feature_concat import (
    DecoderTagOnlyConditioner,
    FeatureConcatConditioner,
    HybridConditioner,
)
from .prepend_prompt import PrependPromptCEConditioner


def _register_default_strategies() -> None:
    ConditionerFactory.register(
        ConditioningStrategy.PREPEND_PROMPT_CE, PrependPromptCEConditioner
    )
    ConditionerFactory.register(
        ConditioningStrategy.DECODER_TAG_ONLY, DecoderTagOnlyConditioner
    )
    ConditionerFactory.register(
        ConditioningStrategy.FEATURE_CONCAT, FeatureConcatConditioner
    )
    ConditionerFactory.register(ConditioningStrategy.HYBRID, HybridConditioner)


_register_default_strategies()


__all__ = [
    "BaseConditioner",
    "ConditionState",
    "AttributeEmbedding",
    "AuxLossScheduler",
    "PrependPromptCEConditioner",
    "DecoderTagOnlyConditioner",
    "FeatureConcatConditioner",
    "HybridConditioner",
]
