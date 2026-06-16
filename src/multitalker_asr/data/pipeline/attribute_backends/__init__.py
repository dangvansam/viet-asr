from typing import Dict, List, Type

from .age import Wav2Vec2AgeBackend
from .base import (
    AttributeBackendError,
    AttributeEnsembler,
    AttributeResult,
    BaseAttributeBackend,
)
from .emotion import FunASREmotionBackend, HuBERTEmotionBackend
from .gender import F0HeuristicGenderBackend, Wav2Vec2GenderBackend
from .language import SpeechBrainLangIDBackend, TagDerivedLanguageBackend
from .region import FormantHeuristicRegionBackend, LoadedRegionClassifier


ATTRIBUTE_REGISTRY: Dict[str, Dict[str, Type[BaseAttributeBackend]]] = {
    "language": {
        "speechbrain_voxlingua": SpeechBrainLangIDBackend,
        "tag_derived": TagDerivedLanguageBackend,
    },
    "emotion": {
        "hubert_superb_er": HuBERTEmotionBackend,
        "funasr_sensevoice": FunASREmotionBackend,
    },
    "gender": {
        "wav2vec2_gender": Wav2Vec2GenderBackend,
        "f0_heuristic": F0HeuristicGenderBackend,
    },
    "age": {
        "wav2vec2_age": Wav2Vec2AgeBackend,
    },
    "region": {
        "formant_heuristic": FormantHeuristicRegionBackend,
        "torch_dialect_clf": LoadedRegionClassifier,
    },
}


def register_attribute_backend(
    axis: str, name: str, cls: Type[BaseAttributeBackend]
) -> None:
    ATTRIBUTE_REGISTRY.setdefault(axis, {})[name] = cls


def build_attribute_backend(axis: str, name: str, **kwargs) -> BaseAttributeBackend:
    if axis not in ATTRIBUTE_REGISTRY:
        raise ValueError(
            f"Unknown axis '{axis}'. Registered: {list(ATTRIBUTE_REGISTRY.keys())}"
        )
    axis_registry = ATTRIBUTE_REGISTRY[axis]
    if name not in axis_registry:
        raise ValueError(
            f"Unknown backend '{name}' for axis '{axis}'. "
            f"Registered: {list(axis_registry.keys())}"
        )
    return axis_registry[name](**kwargs)


def list_attribute_backends(axis: str) -> List[str]:
    return list(ATTRIBUTE_REGISTRY.get(axis, {}).keys())


def list_attribute_axes() -> List[str]:
    return list(ATTRIBUTE_REGISTRY.keys())


__all__ = [
    "BaseAttributeBackend",
    "AttributeBackendError",
    "AttributeResult",
    "AttributeEnsembler",
    "SpeechBrainLangIDBackend",
    "TagDerivedLanguageBackend",
    "HuBERTEmotionBackend",
    "FunASREmotionBackend",
    "Wav2Vec2GenderBackend",
    "F0HeuristicGenderBackend",
    "Wav2Vec2AgeBackend",
    "FormantHeuristicRegionBackend",
    "LoadedRegionClassifier",
    "ATTRIBUTE_REGISTRY",
    "register_attribute_backend",
    "build_attribute_backend",
    "list_attribute_backends",
    "list_attribute_axes",
]
