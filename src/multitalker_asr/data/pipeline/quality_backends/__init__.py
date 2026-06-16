from typing import Dict, List, Type

from .base import BaseQualityBackend, QualityBackendError, QualityResult
from .snr import SNREstimatorBackend


QUALITY_REGISTRY: Dict[str, Type[BaseQualityBackend]] = {
    "snr": SNREstimatorBackend,
}


def register_quality_backend(name: str, cls: Type[BaseQualityBackend]) -> None:
    QUALITY_REGISTRY[name] = cls


def build_quality_backend(name: str, **kwargs) -> BaseQualityBackend:
    if name not in QUALITY_REGISTRY:
        raise ValueError(
            f"Unknown quality backend '{name}'. Registered: {list(QUALITY_REGISTRY.keys())}"
        )
    return QUALITY_REGISTRY[name](**kwargs)


def list_quality_backends() -> List[str]:
    return list(QUALITY_REGISTRY.keys())


__all__ = [
    "BaseQualityBackend",
    "QualityResult",
    "QualityBackendError",
    "SNREstimatorBackend",
    "QUALITY_REGISTRY",
    "register_quality_backend",
    "build_quality_backend",
    "list_quality_backends",
]
