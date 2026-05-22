from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..configs.multitask import MultiTaskConfig


class TaskTokenRegistry:
    """Maps task names to contiguous token ID ranges in a shared embedding table.

    ID layout (default config):
        language:    [0, 1, 2, 3]
        emotion:     [4, 5, 6, 7, 8, 9, 10]
        gender:      [11, 12]
        age:         [13, 14, 15, 16]
        voice_state: [17, 18]
        textnorm:    [19, 20]
    """

    # Human-readable label names per task
    LABEL_NAMES: Dict[str, List[str]] = {
        "language": ["vi", "en", "zh", "auto"],
        "emotion": ["happy", "sad", "angry", "neutral", "fear", "disgust", "surprise"],
        "gender": ["male", "female"],
        "age": ["child", "young", "middle_age", "old"],
        "voice_state": ["sober", "drunk"],
        "textnorm": ["with_itn", "without_itn"],
    }

    def __init__(self, config: MultiTaskConfig):
        self._config = config
        self._task_ranges: OrderedDict[str, tuple] = OrderedDict()
        offset = 0
        for task in config.task_order:
            count = config._class_count_for(task)
            self._task_ranges[task] = (offset, offset + count)
            offset += count
        self._total = offset

    def get_embed_id(self, task: str, class_idx: int) -> int:
        if task not in self._task_ranges:
            raise ValueError(
                f"Unknown task '{task}'. Valid tasks: {list(self._task_ranges.keys())}"
            )
        start, end = self._task_ranges[task]
        if class_idx < 0 or class_idx >= (end - start):
            raise ValueError(
                f"class_idx {class_idx} out of range for task '{task}' "
                f"(valid: 0..{end - start - 1})"
            )
        return start + class_idx

    def get_task_range(self, task: str) -> tuple:
        if task not in self._task_ranges:
            raise ValueError(
                f"Unknown task '{task}'. Valid tasks: {list(self._task_ranges.keys())}"
            )
        return self._task_ranges[task]

    def get_class_count(self, task: str) -> int:
        start, end = self.get_task_range(task)
        return end - start

    def get_label_name(self, task: str, class_idx: int) -> str:
        if task in self.LABEL_NAMES and 0 <= class_idx < len(self.LABEL_NAMES[task]):
            return self.LABEL_NAMES[task][class_idx]
        return f"{task}_{class_idx}"

    def get_class_idx(self, task: str, label_name: str) -> int:
        if task not in self.LABEL_NAMES:
            raise ValueError(f"Unknown task '{task}'")
        try:
            return self.LABEL_NAMES[task].index(label_name)
        except ValueError:
            raise ValueError(
                f"Unknown label '{label_name}' for task '{task}'. "
                f"Valid: {self.LABEL_NAMES[task]}"
            )

    @property
    def total_tokens(self) -> int:
        return self._total

    def task_names(self) -> List[str]:
        return list(self._task_ranges.keys())


class PromptEmbedding(nn.Module):
    """SenseVoice-style prompt token embedding.

    Prepends learned embeddings for each paralinguistic task to speech features.
    Each task position gets one embedding looked up from a shared table.

    Reference: funasr/models/sense_voice/model.py:642-648, 744-774
    """

    def __init__(self, config: MultiTaskConfig):
        super().__init__()
        self._config = config
        self._registry = TaskTokenRegistry(config)
        self.embed = nn.Embedding(self._registry.total_tokens, config.prompt_embed_dim)

    @property
    def registry(self) -> TaskTokenRegistry:
        return self._registry

    @property
    def num_positions(self) -> int:
        return self._config.num_prompt_positions

    def forward(
        self, task_labels: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Create prompt embeddings from task labels.

        Args:
            task_labels: {task_name: Tensor[B]} with class indices per task.
                         Missing tasks default to class index 0.

        Returns:
            Tensor [B, num_prompt_positions, embed_dim]
        """
        if task_labels is None:
            task_labels = {}

        # Infer batch size from any provided tensor
        batch_size = None
        for t in task_labels.values():
            batch_size = t.shape[0]
            break
        if batch_size is None:
            raise RuntimeError(
                "Cannot infer batch size: task_labels is empty. "
                "Provide at least one task tensor."
            )

        embeddings = []

        for task in self._config.task_order:
            if task in task_labels:
                class_indices = task_labels[task]
            else:
                class_indices = torch.zeros(
                    batch_size, dtype=torch.long, device=self.embed.weight.device
                )

            start, _ = self._registry.get_task_range(task)
            embed_ids = class_indices + start
            task_embed = self.embed(embed_ids)  # [B, embed_dim]
            embeddings.append(task_embed.unsqueeze(1))  # [B, 1, embed_dim]

        return torch.cat(embeddings, dim=1)  # [B, num_positions, embed_dim]
