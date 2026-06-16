from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .base import BaseMetric


@dataclass
class AttributeConfusion:
    axis: str
    label_space: List[str]
    matrix: Dict[Tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    total: int = 0
    correct: int = 0

    def update(self, reference: str, prediction: str) -> None:
        self.matrix[(reference, prediction)] += 1
        self.total += 1
        if reference == prediction:
            self.correct += 1

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def per_label_f1(self) -> Dict[str, float]:
        f1: Dict[str, float] = {}
        for label in self.label_space:
            tp = self.matrix.get((label, label), 0)
            fp = sum(
                count
                for (ref, hyp), count in self.matrix.items()
                if hyp == label and ref != label
            )
            fn = sum(
                count
                for (ref, hyp), count in self.matrix.items()
                if ref == label and hyp != label
            )
            denom = 2 * tp + fp + fn
            f1[label] = 2 * tp / denom if denom > 0 else 0.0
        return f1

    def macro_f1(self) -> float:
        per_label = self.per_label_f1()
        if not per_label:
            return 0.0
        return sum(per_label.values()) / len(per_label)

    def precision_recall(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for label in self.label_space:
            tp = self.matrix.get((label, label), 0)
            fp = sum(
                count
                for (ref, hyp), count in self.matrix.items()
                if hyp == label and ref != label
            )
            fn = sum(
                count
                for (ref, hyp), count in self.matrix.items()
                if ref == label and hyp != label
            )
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            out[label] = {"precision": precision, "recall": recall}
        return out


class AttributeAccuracyMetric(BaseMetric):
    def __init__(self, label_space: List[str], axis: str = "attribute"):
        self._label_space = list(label_space)
        self._axis = axis

    def compute(self, reference: Any, hypothesis: Any) -> Dict[str, float]:
        if reference is None or hypothesis is None:
            return {f"{self._axis}_accuracy": 0.0, f"{self._axis}_total": 0.0}
        match = 1.0 if reference == hypothesis else 0.0
        return {f"{self._axis}_accuracy": match, f"{self._axis}_total": 1.0}

    def aggregate(self, per_file_results: List[Dict[str, float]]) -> Dict[str, float]:
        total = sum(r.get(f"{self._axis}_total", 0.0) for r in per_file_results)
        correct = sum(r.get(f"{self._axis}_accuracy", 0.0) for r in per_file_results)
        accuracy = correct / total if total > 0 else 0.0
        return {f"{self._axis}_accuracy": accuracy, f"{self._axis}_samples": total}


class AttributeMacroF1Metric(BaseMetric):
    def __init__(self, label_space: List[str], axis: str = "attribute"):
        self._label_space = list(label_space)
        self._axis = axis
        self._confusion = AttributeConfusion(axis=axis, label_space=self._label_space)

    @property
    def confusion(self) -> AttributeConfusion:
        return self._confusion

    def reset(self) -> None:
        self._confusion = AttributeConfusion(
            axis=self._axis, label_space=self._label_space
        )

    def compute(self, reference: Any, hypothesis: Any) -> Dict[str, float]:
        if reference is None or hypothesis is None:
            return {f"{self._axis}_recorded": 0.0}
        self._confusion.update(str(reference), str(hypothesis))
        return {f"{self._axis}_recorded": 1.0}

    def aggregate(self, per_file_results: List[Dict[str, float]]) -> Dict[str, float]:
        out = {
            f"{self._axis}_accuracy": self._confusion.accuracy,
            f"{self._axis}_macro_f1": self._confusion.macro_f1(),
            f"{self._axis}_samples": self._confusion.total,
        }
        return out


class AttributePipelineEvaluator:
    """Evaluate predictions for all attribute axes in one pass."""

    def __init__(self, axes_with_label_space: Dict[str, List[str]]):
        self._metrics: Dict[str, AttributeMacroF1Metric] = {
            axis: AttributeMacroF1Metric(label_space=labels, axis=axis)
            for axis, labels in axes_with_label_space.items()
        }

    @property
    def axes(self) -> List[str]:
        return list(self._metrics.keys())

    def reset(self) -> None:
        for metric in self._metrics.values():
            metric.reset()

    def update(
        self,
        references: Dict[str, Optional[str]],
        predictions: Dict[str, Optional[str]],
    ) -> None:
        for axis, metric in self._metrics.items():
            metric.compute(references.get(axis), predictions.get(axis))

    def update_batch(
        self,
        pairs: Iterable[Tuple[Dict[str, Optional[str]], Dict[str, Optional[str]]]],
    ) -> None:
        for ref, hyp in pairs:
            self.update(ref, hyp)

    def report(self) -> Dict[str, Dict[str, float]]:
        report: Dict[str, Dict[str, float]] = {}
        for axis, metric in self._metrics.items():
            report[axis] = metric.aggregate([])
            report[axis]["per_label_f1"] = metric.confusion.per_label_f1()
            report[axis]["precision_recall"] = metric.confusion.precision_recall()
        return report

    def average_macro_f1(self) -> float:
        if not self._metrics:
            return 0.0
        scores = [m.confusion.macro_f1() for m in self._metrics.values()]
        return sum(scores) / len(scores)
