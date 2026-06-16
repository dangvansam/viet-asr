from typing import Dict, List, Type

from .base import (
    AlignBackendError,
    AlignedWord,
    AlignResult,
    BaseAlignBackend,
    alignment_score,
)
from .funasr_align import FunASRAlignBackend
from .funasr_nano_align import FunASRNanoAlignBackend
from .mms_fa import MMSAlignBackend
from .nemo_nfa import NeMoNFABackend
from .qwen3 import Qwen3AlignBackend
from .qwen3_service import Qwen3ServiceAlignBackend
from .service import ServiceAlignBackend


ALIGN_REGISTRY: Dict[str, Type[BaseAlignBackend]] = {
    "qwen3": Qwen3AlignBackend,
    "qwen3_service": Qwen3ServiceAlignBackend,
    "nemo_nfa": NeMoNFABackend,
    "mms_fa": MMSAlignBackend,
    "service": ServiceAlignBackend,
    "funasr_align": FunASRAlignBackend,
    "funasr_nano_align": FunASRNanoAlignBackend,
}


def register_align_backend(name: str, cls: Type[BaseAlignBackend]) -> None:
    ALIGN_REGISTRY[name] = cls


def build_align_backend(name: str, **kwargs) -> BaseAlignBackend:
    if name not in ALIGN_REGISTRY:
        raise ValueError(
            f"Unknown align backend '{name}'. Registered: {list(ALIGN_REGISTRY.keys())}"
        )
    return ALIGN_REGISTRY[name](**kwargs)


def list_align_backends() -> List[str]:
    return list(ALIGN_REGISTRY.keys())


__all__ = [
    "AlignBackendError",
    "AlignedWord",
    "AlignResult",
    "BaseAlignBackend",
    "alignment_score",
    "Qwen3AlignBackend",
    "Qwen3ServiceAlignBackend",
    "NeMoNFABackend",
    "MMSAlignBackend",
    "ServiceAlignBackend",
    "FunASRAlignBackend",
    "FunASRNanoAlignBackend",
    "ALIGN_REGISTRY",
    "register_align_backend",
    "build_align_backend",
    "list_align_backends",
]
