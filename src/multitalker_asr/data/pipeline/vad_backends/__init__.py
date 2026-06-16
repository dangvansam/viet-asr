from typing import Dict, List, Type

from .base import (
    BaseVADBackend,
    VADBackendError,
    VADResult,
    VADSegment,
    assemble_frames,
    merge_segments,
)
from .consensus import ConsensusVADBackend
from .fsmn_vad import FsmnVADBackend
from .pyannote_seg import PyannoteSegmentationVADBackend
from .service import ServiceVADBackend
from .silero import SileroVADBackend
from .ten_vad import TenVADBackend


VAD_REGISTRY: Dict[str, Type[BaseVADBackend]] = {
    "silero": SileroVADBackend,
    "fsmn": FsmnVADBackend,
    "ten": TenVADBackend,
    "pyannote_seg": PyannoteSegmentationVADBackend,
    "consensus": ConsensusVADBackend,
    "service": ServiceVADBackend,
}


def register_vad_backend(name: str, cls: Type[BaseVADBackend]) -> None:
    VAD_REGISTRY[name] = cls


def build_vad_backend(name: str, **kwargs) -> BaseVADBackend:
    if name not in VAD_REGISTRY:
        raise ValueError(
            f"Unknown VAD backend '{name}'. Registered: {list(VAD_REGISTRY.keys())}"
        )
    return VAD_REGISTRY[name](**kwargs)


def list_vad_backends() -> List[str]:
    return list(VAD_REGISTRY.keys())


__all__ = [
    "BaseVADBackend",
    "VADBackendError",
    "VADResult",
    "VADSegment",
    "assemble_frames",
    "merge_segments",
    "SileroVADBackend",
    "FsmnVADBackend",
    "TenVADBackend",
    "PyannoteSegmentationVADBackend",
    "ConsensusVADBackend",
    "ServiceVADBackend",
    "VAD_REGISTRY",
    "register_vad_backend",
    "build_vad_backend",
    "list_vad_backends",
]
