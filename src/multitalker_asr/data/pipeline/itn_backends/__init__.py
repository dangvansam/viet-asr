from typing import Dict, List, Type

from .base import BaseITNBackend, ITNBackendError, ITNResult
from .funasr_itn import FunASRITN
from .llm_itn import LLMITNBackend
from .nemo_itn import NeMoTextNorm
from .strategies import (
    BaseITNStrategy,
    FirstSuccessStrategy,
    ITNStrategyFactory,
    LLMJudgeITNStrategy,
    VoteITNStrategy,
)
from .vietasr_itn import VietASRITN


ITN_REGISTRY: Dict[str, Type[BaseITNBackend]] = {
    "llm_itn": LLMITNBackend,
    "funasr_itn": FunASRITN,
    "vietasr_itn": VietASRITN,
    "nemo_itn": NeMoTextNorm,
}


def register_itn_backend(name: str, cls: Type[BaseITNBackend]) -> None:
    ITN_REGISTRY[name] = cls


def build_itn_backend(name: str, **kwargs) -> BaseITNBackend:
    if name not in ITN_REGISTRY:
        raise ValueError(
            f"Unknown ITN backend '{name}'. Registered: {list(ITN_REGISTRY.keys())}"
        )
    return ITN_REGISTRY[name](**kwargs)


def list_itn_backends() -> List[str]:
    return list(ITN_REGISTRY.keys())


__all__ = [
    "BaseITNBackend",
    "ITNBackendError",
    "ITNResult",
    "BaseITNStrategy",
    "FirstSuccessStrategy",
    "VoteITNStrategy",
    "LLMJudgeITNStrategy",
    "ITNStrategyFactory",
    "LLMITNBackend",
    "FunASRITN",
    "VietASRITN",
    "NeMoTextNorm",
    "ITN_REGISTRY",
    "register_itn_backend",
    "build_itn_backend",
    "list_itn_backends",
]
