from dataclasses import dataclass, field
from typing import Any, Dict, List

from .base import BaseConfig


@dataclass
class AttributeBackendConfig(BaseConfig):
    name: str = ""
    enabled: bool = True
    device: str = "cpu"
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttributeAxisConfig(BaseConfig):
    axis: str = ""
    backends: List[AttributeBackendConfig] = field(default_factory=list)
    strategy: str = "vote"
    min_confidence: float = 0.0
    enabled: bool = True

    def __post_init__(self):
        if not self.axis:
            raise ValueError("AttributeAxisConfig.axis must be non-empty")
        if self.strategy not in ("vote", "mean_posterior", "first"):
            raise ValueError(
                f"Unknown strategy '{self.strategy}'. "
                f"Valid: vote, mean_posterior, first"
            )
        if self.min_confidence < 0.0 or self.min_confidence > 1.0:
            raise ValueError(
                f"min_confidence must be in [0,1], got {self.min_confidence}"
            )

    def enabled_backends(self) -> List[AttributeBackendConfig]:
        return [b for b in self.backends if b.enabled]


@dataclass
class AttributePipelineConfig(BaseConfig):
    axes: List[AttributeAxisConfig] = field(default_factory=list)

    def __post_init__(self):
        seen = set()
        for axis_cfg in self.axes:
            if axis_cfg.axis in seen:
                raise ValueError(f"Duplicate axis '{axis_cfg.axis}' in pipeline")
            seen.add(axis_cfg.axis)

    def enabled_axes(self) -> List[AttributeAxisConfig]:
        return [a for a in self.axes if a.enabled]

    @classmethod
    def default_full(cls) -> "AttributePipelineConfig":
        return cls(
            axes=[
                AttributeAxisConfig(
                    axis="language",
                    backends=[
                        AttributeBackendConfig(name="tag_derived"),
                        AttributeBackendConfig(name="speechbrain_voxlingua"),
                    ],
                    strategy="vote",
                ),
                AttributeAxisConfig(
                    axis="emotion",
                    backends=[
                        AttributeBackendConfig(name="funasr_sensevoice"),
                        AttributeBackendConfig(name="hubert_superb_er"),
                    ],
                    strategy="mean_posterior",
                ),
                AttributeAxisConfig(
                    axis="gender",
                    backends=[
                        AttributeBackendConfig(name="wav2vec2_gender"),
                        AttributeBackendConfig(name="f0_heuristic"),
                    ],
                    strategy="vote",
                ),
                AttributeAxisConfig(
                    axis="age",
                    backends=[AttributeBackendConfig(name="wav2vec2_age")],
                    strategy="first",
                ),
                AttributeAxisConfig(
                    axis="region",
                    backends=[AttributeBackendConfig(name="formant_heuristic")],
                    strategy="first",
                ),
            ]
        )
