from typing import Union

from omegaconf import OmegaConf

from .model import ModelConfig
from .training import TrainingConfig
from .inference import InferenceConfig
from .data import DataConfig
from .eval import EvalConfig


CONFIG_TYPES = {
    "model": ModelConfig,
    "training": TrainingConfig,
    "inference": InferenceConfig,
    "data": DataConfig,
    "eval": EvalConfig,
}


def get_config(
    config_type: str = "inference",
    **kwargs,
) -> Union[ModelConfig, TrainingConfig, InferenceConfig, DataConfig, EvalConfig]:
    if config_type not in CONFIG_TYPES:
        raise ValueError(f"Unknown config type: {config_type}")

    config_class = CONFIG_TYPES[config_type]
    base_cfg = OmegaConf.structured(config_class())
    overrides = OmegaConf.create(kwargs)
    return OmegaConf.merge(base_cfg, overrides)
