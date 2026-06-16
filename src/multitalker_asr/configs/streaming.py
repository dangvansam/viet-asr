from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .base import BaseConfig


FRAME_DURATION_MS = 80.0
DEFAULT_LEFT_CONTEXT = 56


class ChunkPreset(str, Enum):
    """Nemotron-style streaming operating points. Frame = 80 ms."""

    CHUNK_80MS = "chunk_80ms"
    CHUNK_160MS = "chunk_160ms"
    CHUNK_320MS = "chunk_320ms"
    CHUNK_560MS = "chunk_560ms"
    CHUNK_1120MS = "chunk_1120ms"


_PRESET_RIGHT_CONTEXT: Dict[ChunkPreset, int] = {
    ChunkPreset.CHUNK_80MS: 0,
    ChunkPreset.CHUNK_160MS: 1,
    ChunkPreset.CHUNK_320MS: 3,
    ChunkPreset.CHUNK_560MS: 6,
    ChunkPreset.CHUNK_1120MS: 13,
}


def preset_to_att_context(
    preset: ChunkPreset, left_context: int = DEFAULT_LEFT_CONTEXT
) -> List[int]:
    return [int(left_context), _PRESET_RIGHT_CONTEXT[preset]]


def preset_chunk_ms(preset: ChunkPreset) -> float:
    right = _PRESET_RIGHT_CONTEXT[preset]
    return (1 + right) * FRAME_DURATION_MS


def all_preset_att_contexts(
    left_context: int = DEFAULT_LEFT_CONTEXT,
) -> List[Tuple[ChunkPreset, List[int]]]:
    return [(p, preset_to_att_context(p, left_context)) for p in ChunkPreset]


@dataclass
class StreamingProfile(BaseConfig):
    """Runtime-switchable streaming profile."""

    left_context: int = DEFAULT_LEFT_CONTEXT
    preset: ChunkPreset = ChunkPreset.CHUNK_1120MS
    available_presets: List[ChunkPreset] = field(
        default_factory=lambda: list(ChunkPreset)
    )
    auto_select_for_latency_ms: Optional[float] = None

    def __post_init__(self):
        if isinstance(self.preset, str):
            self.preset = ChunkPreset(self.preset)
        if self.left_context < 0:
            raise ValueError(f"left_context must be >= 0, got {self.left_context}")
        if self.preset not in self.available_presets:
            raise ValueError(
                f"preset {self.preset} not in available_presets "
                f"({[p.value for p in self.available_presets]})"
            )
        if self.auto_select_for_latency_ms is not None and self.auto_select_for_latency_ms <= 0:
            raise ValueError(
                f"auto_select_for_latency_ms must be > 0 if set, "
                f"got {self.auto_select_for_latency_ms}"
            )

    @property
    def att_context_size(self) -> List[int]:
        return preset_to_att_context(self.preset, self.left_context)

    @property
    def chunk_ms(self) -> float:
        return preset_chunk_ms(self.preset)

    def with_preset(self, preset: ChunkPreset) -> "StreamingProfile":
        if preset not in self.available_presets:
            raise ValueError(
                f"preset {preset} not in available_presets "
                f"({[p.value for p in self.available_presets]})"
            )
        return StreamingProfile(
            left_context=self.left_context,
            preset=preset,
            available_presets=list(self.available_presets),
            auto_select_for_latency_ms=self.auto_select_for_latency_ms,
        )

    def select_for_latency_budget(self, latency_ms: float) -> ChunkPreset:
        if latency_ms <= 0:
            raise ValueError(f"latency_ms must be > 0, got {latency_ms}")
        candidates = sorted(
            self.available_presets, key=preset_chunk_ms, reverse=True
        )
        for preset in candidates:
            if preset_chunk_ms(preset) <= latency_ms:
                return preset
        return min(self.available_presets, key=preset_chunk_ms)
