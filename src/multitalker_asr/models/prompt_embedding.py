import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..configs.attribute_vocab import DEFAULT_ATTRIBUTE_VOCABULARY, AttributeVocabulary
from ..configs.multitask import MultiTaskConfig


class TaskTokenRegistry:
    """Maps task names to contiguous token ID ranges + canonical decoder tag strings.

    Two-layer mapping:
        embed_id  ── shared embedding table index (input-side, prepend prompt)
        tag       ── decoder vocabulary special token string (output-side, e.g. "<vi-VN>")
    """

    LEGACY_LABEL_NAMES: Dict[str, List[str]] = {
        "language": ["vi", "en", "zh", "auto"],
        "emotion": ["happy", "sad", "angry", "neutral", "fear", "disgust", "surprise"],
        "gender": ["male", "female"],
        "age": ["child", "young", "middle_age", "old"],
        "voice_state": ["sober", "drunk"],
        "textnorm": ["with_itn", "without_itn"],
        "region": ["northern", "central", "southern"],
    }

    _TAG_PATTERN = re.compile(r"<[^<>\s]+>")

    def __init__(
        self,
        config: MultiTaskConfig,
        vocabulary: Optional[AttributeVocabulary] = None,
        prefer_legacy_labels: bool = True,
    ):
        self._config = config
        self._vocab = vocabulary if vocabulary is not None else DEFAULT_ATTRIBUTE_VOCABULARY
        self._prefer_legacy = prefer_legacy_labels

        self._task_ranges: "OrderedDict[str, Tuple[int, int]]" = OrderedDict()
        offset = 0
        for task in config.task_order:
            count = config._class_count_for(task)
            self._task_ranges[task] = (offset, offset + count)
            offset += count
        self._total = offset

        self._label_names: Dict[str, List[str]] = {}
        for task in config.task_order:
            labels = self._derive_label_names(task)
            self._label_names[task] = labels

    def _derive_label_names(self, task: str) -> List[str]:
        count = self._config._class_count_for(task)
        legacy = self.LEGACY_LABEL_NAMES.get(task)
        vocab_labels: Optional[List[str]] = None
        try:
            vocab_labels = list(self._vocab.axis_by_name(task).labels)
        except ValueError:
            vocab_labels = None

        primary = legacy if self._prefer_legacy and legacy is not None else vocab_labels
        fallback = vocab_labels if primary is legacy else legacy

        for candidate in (primary, fallback):
            if candidate is None:
                continue
            if count <= len(candidate):
                return candidate[:count]
            return candidate + [f"{task}_{i}" for i in range(len(candidate), count)]

        return [f"{task}_{i}" for i in range(count)]

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

    def get_task_range(self, task: str) -> Tuple[int, int]:
        if task not in self._task_ranges:
            raise ValueError(
                f"Unknown task '{task}'. Valid tasks: {list(self._task_ranges.keys())}"
            )
        return self._task_ranges[task]

    def get_class_count(self, task: str) -> int:
        start, end = self.get_task_range(task)
        return end - start

    def get_label_name(self, task: str, class_idx: int) -> str:
        labels = self._label_names.get(task)
        if labels is None or not (0 <= class_idx < len(labels)):
            return f"{task}_{class_idx}"
        return labels[class_idx]

    def get_class_idx(self, task: str, label_name: str) -> int:
        labels = self._label_names.get(task)
        if labels is None:
            raise ValueError(f"Unknown task '{task}'")
        try:
            return labels.index(label_name)
        except ValueError:
            raise ValueError(
                f"Unknown label '{label_name}' for task '{task}'. Valid: {labels}"
            )

    def tag(self, task: str, class_idx: int) -> str:
        label = self.get_label_name(task, class_idx)
        try:
            axis = self._vocab.axis_by_name(task)
            return axis.tag(label)
        except ValueError:
            return f"<{task}:{label}>"

    def tag_from_label(self, task: str, label: str) -> str:
        try:
            axis = self._vocab.axis_by_name(task)
            return axis.tag(label)
        except ValueError:
            return f"<{task}:{label}>"

    def all_tag_strings(self) -> List[str]:
        tags: List[str] = []
        for task in self._task_ranges:
            count = self.get_class_count(task)
            for idx in range(count):
                tags.append(self.tag(task, idx))
        return tags

    def parse_tag(self, tag: str) -> Optional[Tuple[str, int]]:
        result = self._vocab.tag_to_axis_label(tag)
        if result is None:
            return None
        task, label = result
        if task not in self._task_ranges:
            return None
        try:
            class_idx = self.get_class_idx(task, label)
        except ValueError:
            return None
        return (task, class_idx)

    def parse_tags_from_text(self, text: str) -> Tuple[str, Dict[str, int]]:
        attributes: Dict[str, int] = {}
        for match in self._TAG_PATTERN.findall(text):
            parsed = self.parse_tag(match)
            if parsed is not None:
                task, class_idx = parsed
                attributes[task] = class_idx
        stripped = self._TAG_PATTERN.sub("", text)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        return stripped, attributes

    def append_tags(self, text: str, attributes: Dict[str, int]) -> str:
        tag_str = "".join(
            self.tag(task, attributes[task])
            for task in self._task_ranges
            if task in attributes
        )
        return f"{text}{tag_str}" if tag_str else text

    @property
    def total_tokens(self) -> int:
        return self._total

    def task_names(self) -> List[str]:
        return list(self._task_ranges.keys())

    @property
    def vocabulary(self) -> AttributeVocabulary:
        return self._vocab


class PromptEmbedding(nn.Module):
    """SenseVoice-style prompt token embedding.

    Prepends learned embeddings for each paralinguistic task to speech features.
    Each task position gets one embedding looked up from a shared table.
    """

    def __init__(
        self,
        config: MultiTaskConfig,
        vocabulary: Optional[AttributeVocabulary] = None,
        prefer_legacy_labels: bool = True,
    ):
        super().__init__()
        self._config = config
        self._registry = TaskTokenRegistry(
            config,
            vocabulary=vocabulary,
            prefer_legacy_labels=prefer_legacy_labels,
        )
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
        if task_labels is None:
            task_labels = {}

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
            task_embed = self.embed(embed_ids)
            embeddings.append(task_embed.unsqueeze(1))

        return torch.cat(embeddings, dim=1)
