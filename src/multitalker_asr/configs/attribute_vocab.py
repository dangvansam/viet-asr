from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .base import BaseConfig


@dataclass(frozen=True)
class AttributeAxis:
    name: str
    labels: Tuple[str, ...]
    tag_prefix: str = ""

    def index_of(self, label: str) -> int:
        try:
            return self.labels.index(label)
        except ValueError:
            raise ValueError(
                f"Unknown label '{label}' for axis '{self.name}'. "
                f"Valid: {list(self.labels)}"
            )

    def label_of(self, index: int) -> str:
        if not 0 <= index < len(self.labels):
            raise ValueError(
                f"Index {index} out of range for axis '{self.name}' (0..{len(self.labels) - 1})"
            )
        return self.labels[index]

    def tag(self, label: str) -> str:
        return f"<{self.tag_prefix}{label}>"

    def num_classes(self) -> int:
        return len(self.labels)


@dataclass
class AttributeVocabulary(BaseConfig):
    language: AttributeAxis = field(
        default_factory=lambda: AttributeAxis(
            name="language",
            labels=("vi-VN", "en-US", "zh-CN", "auto"),
            tag_prefix="",
        )
    )
    emotion: AttributeAxis = field(
        default_factory=lambda: AttributeAxis(
            name="emotion",
            labels=("neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"),
            tag_prefix="emo:",
        )
    )
    gender: AttributeAxis = field(
        default_factory=lambda: AttributeAxis(
            name="gender",
            labels=("male", "female"),
            tag_prefix="gen:",
        )
    )
    age: AttributeAxis = field(
        default_factory=lambda: AttributeAxis(
            name="age",
            labels=("child", "teen", "adult", "senior"),
            tag_prefix="age:",
        )
    )
    region: AttributeAxis = field(
        default_factory=lambda: AttributeAxis(
            name="region",
            labels=("northern", "central", "southern"),
            tag_prefix="reg:",
        )
    )

    def axes(self) -> List[AttributeAxis]:
        return [self.language, self.emotion, self.gender, self.age, self.region]

    def axis_by_name(self, name: str) -> AttributeAxis:
        for axis in self.axes():
            if axis.name == name:
                return axis
        raise ValueError(
            f"Unknown axis '{name}'. Valid: {[a.name for a in self.axes()]}"
        )

    def all_tag_strings(self) -> List[str]:
        tags: List[str] = []
        for axis in self.axes():
            for label in axis.labels:
                tags.append(axis.tag(label))
        return tags

    def total_classes(self) -> int:
        return sum(axis.num_classes() for axis in self.axes())

    def tag_to_axis_label(self, tag: str) -> Optional[Tuple[str, str]]:
        if not (tag.startswith("<") and tag.endswith(">")):
            return None
        body = tag[1:-1]
        for axis in self.axes():
            if axis.tag_prefix and body.startswith(axis.tag_prefix):
                label = body[len(axis.tag_prefix) :]
                if label in axis.labels:
                    return (axis.name, label)
            elif not axis.tag_prefix and body in axis.labels:
                return (axis.name, body)
        return None

    def class_counts(self) -> Dict[str, int]:
        return {axis.name: axis.num_classes() for axis in self.axes()}


DEFAULT_ATTRIBUTE_VOCABULARY = AttributeVocabulary()
