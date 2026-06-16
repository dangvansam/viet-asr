from dataclasses import dataclass
from typing import Any, List, Optional

from loguru import logger

from ..configs.streaming import (
    ChunkPreset,
    StreamingProfile,
    all_preset_att_contexts,
    preset_to_att_context,
)


@dataclass
class CacheAwareCapability:
    has_streaming_cfg: bool
    has_set_att_context: bool
    has_cache_last_channel: bool
    has_cache_last_time: bool
    chunk_size: Optional[int] = None
    cache_drop_size: Optional[int] = None

    @property
    def is_cache_aware(self) -> bool:
        return self.has_streaming_cfg and self.has_set_att_context

    def summary(self) -> str:
        if not self.is_cache_aware:
            return "NOT cache-aware (missing streaming_cfg or set_default_att_context_size)"
        details = []
        if self.chunk_size is not None:
            details.append(f"chunk_size={self.chunk_size}")
        if self.cache_drop_size is not None:
            details.append(f"cache_drop={self.cache_drop_size}")
        details.append(f"cache_last_channel={self.has_cache_last_channel}")
        details.append(f"cache_last_time={self.has_cache_last_time}")
        return "cache-aware: " + ", ".join(details)


class CacheAwareValidator:
    """Probe a NeMo ASR model for cache-aware streaming capabilities."""

    def inspect(self, asr_model: Any) -> CacheAwareCapability:
        encoder = getattr(asr_model, "encoder", None)
        if encoder is None:
            return CacheAwareCapability(False, False, False, False)

        streaming_cfg = getattr(encoder, "streaming_cfg", None)
        has_setter = hasattr(encoder, "set_default_att_context_size")
        cache_channel = hasattr(encoder, "cache_last_channel") or hasattr(
            encoder, "_cache_last_channel"
        )
        cache_time = hasattr(encoder, "cache_last_time") or hasattr(
            encoder, "_cache_last_time"
        )

        chunk = None
        drop = None
        if streaming_cfg is not None:
            chunk = getattr(streaming_cfg, "chunk_size", None)
            drop = getattr(streaming_cfg, "drop_extra_pre_encoded", None)
            if drop is None:
                drop = getattr(streaming_cfg, "cache_drop_size", None)

        return CacheAwareCapability(
            has_streaming_cfg=streaming_cfg is not None,
            has_set_att_context=has_setter,
            has_cache_last_channel=cache_channel,
            has_cache_last_time=cache_time,
            chunk_size=chunk,
            cache_drop_size=drop,
        )


class ChunkSwitcher:
    """Runtime switch between Nemotron operating points without retraining."""

    def __init__(self, asr_model: Any, profile: Optional[StreamingProfile] = None):
        self._asr_model = asr_model
        self._profile = profile or StreamingProfile()
        self._current_preset: Optional[ChunkPreset] = None
        self._validator = CacheAwareValidator()

    @property
    def profile(self) -> StreamingProfile:
        return self._profile

    @property
    def current_preset(self) -> Optional[ChunkPreset]:
        return self._current_preset

    def validate(self) -> CacheAwareCapability:
        capability = self._validator.inspect(self._asr_model)
        if not capability.is_cache_aware:
            logger.warning(f"ChunkSwitcher: {capability.summary()}")
        else:
            logger.info(f"ChunkSwitcher: {capability.summary()}")
        return capability

    def apply(self, preset: ChunkPreset) -> List[int]:
        if preset not in self._profile.available_presets:
            raise ValueError(
                f"preset {preset.value} not in available presets "
                f"({[p.value for p in self._profile.available_presets]})"
            )
        att_context = preset_to_att_context(preset, self._profile.left_context)
        encoder = getattr(self._asr_model, "encoder", None)
        if encoder is None or not hasattr(encoder, "set_default_att_context_size"):
            raise RuntimeError(
                "asr_model.encoder does not support set_default_att_context_size"
            )
        encoder.set_default_att_context_size(att_context_size=att_context)
        self._current_preset = preset
        self._profile = self._profile.with_preset(preset)
        logger.info(
            f"Applied chunk preset {preset.value} "
            f"(att_context={att_context}, chunk_ms={self._profile.chunk_ms:.0f})"
        )
        return att_context

    def apply_for_latency(self, latency_ms: float) -> ChunkPreset:
        preset = self._profile.select_for_latency_budget(latency_ms)
        self.apply(preset)
        return preset

    def enumerate_operating_points(self) -> List[tuple]:
        return all_preset_att_contexts(self._profile.left_context)
