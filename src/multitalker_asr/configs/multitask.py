from dataclasses import dataclass, field
from typing import List

from .base import BaseConfig


@dataclass
class MultiTaskConfig(BaseConfig):
    """Configuration for multi-task paralinguistic learning (SenseVoice-style prompt tokens)."""

    num_prompt_positions: int = 6
    prompt_embed_dim: int = 80
    emotion_classes: int = 7
    gender_classes: int = 2
    age_classes: int = 4
    voice_state_classes: int = 2
    language_classes: int = 4
    textnorm_classes: int = 2
    ce_loss_weight: float = 1.0
    encoder_source: str = "nemo"

    # Ordered task names matching prompt positions
    task_order: List[str] = field(
        default_factory=lambda: [
            "language",
            "emotion",
            "gender",
            "age",
            "voice_state",
            "textnorm",
        ]
    )

    def __post_init__(self):
        if self.prompt_embed_dim <= 0:
            raise ValueError(f"prompt_embed_dim must be > 0, got {self.prompt_embed_dim}")
        for task in self.task_order:
            count = self._class_count_for(task)
            if count <= 0:
                raise ValueError(f"Class count for '{task}' must be > 0, got {count}")
        if self.encoder_source not in ("nemo", "funasr", "scratch"):
            raise ValueError(
                f"encoder_source must be 'nemo', 'funasr', or 'scratch', got '{self.encoder_source}'"
            )
        if len(self.task_order) != self.num_prompt_positions:
            raise ValueError(
                f"task_order length ({len(self.task_order)}) must match "
                f"num_prompt_positions ({self.num_prompt_positions})"
            )

    def _class_count_for(self, task: str) -> int:
        mapping = {
            "emotion": self.emotion_classes,
            "gender": self.gender_classes,
            "age": self.age_classes,
            "voice_state": self.voice_state_classes,
            "language": self.language_classes,
            "textnorm": self.textnorm_classes,
        }
        if task not in mapping:
            raise ValueError(f"Unknown task '{task}'. Valid: {list(mapping.keys())}")
        return mapping[task]

    @property
    def total_prompt_tokens(self) -> int:
        return sum(self._class_count_for(t) for t in self.task_order)

    @property
    def task_class_counts(self) -> dict:
        return {t: self._class_count_for(t) for t in self.task_order}
