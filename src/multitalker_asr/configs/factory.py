from typing import Union

from omegaconf import OmegaConf

from .conditioning import ConditioningConfig
from .data import DataConfig
from .eval import EvalConfig
from .google_speech import GoogleSpeechConfig
from .inference import InferenceConfig
from .model import ModelConfig
from .multitask import MultiTaskConfig
from .training import TrainingConfig


CONFIG_TYPES = {
    "model": ModelConfig,
    "training": TrainingConfig,
    "inference": InferenceConfig,
    "data": DataConfig,
    "eval": EvalConfig,
    "multitask": MultiTaskConfig,
    "conditioning": ConditioningConfig,
    "google_speech": GoogleSpeechConfig,
}


def get_config(
    config_type: str = "inference",
    **kwargs,
) -> Union[
    ModelConfig,
    TrainingConfig,
    InferenceConfig,
    DataConfig,
    EvalConfig,
    MultiTaskConfig,
    ConditioningConfig,
    GoogleSpeechConfig,
]:
    if config_type not in CONFIG_TYPES:
        raise ValueError(f"Unknown config type: {config_type}")

    config_class = CONFIG_TYPES[config_type]
    base_cfg = OmegaConf.structured(config_class())
    overrides = OmegaConf.create(kwargs)
    return OmegaConf.merge(base_cfg, overrides)
