from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class AttributeResult:
    axis: str
    label: str
    confidence: float = 1.0
    posterior: Optional[Dict[str, float]] = None
    backend: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


class BaseAttributeBackend(ABC):
    axis: str = ""
    name: str = ""
    label_space: List[str] = []

    @abstractmethod
    def load(self, device: str = "cpu") -> None:
        ...

    @abstractmethod
    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int,
        text: Optional[str] = None,
    ) -> AttributeResult:
        ...

    def unload(self) -> None:
        return None

    @property
    def is_loaded(self) -> bool:
        return getattr(self, "_loaded", False)


class AttributeBackendError(RuntimeError):
    pass


class AttributeEnsembler:
    def __init__(
        self,
        backends: List[BaseAttributeBackend],
        axis: str,
        strategy: str = "vote",
    ):
        if not backends:
            raise ValueError("AttributeEnsembler requires at least one backend")
        self._backends = backends
        self._axis = axis
        self._strategy = strategy.lower()
        if self._strategy not in ("vote", "mean_posterior", "first"):
            raise ValueError(
                f"Unknown attribute strategy '{strategy}'. "
                f"Valid: vote, mean_posterior, first"
            )

    @property
    def axis(self) -> str:
        return self._axis

    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int,
        text: Optional[str] = None,
    ) -> AttributeResult:
        results: List[AttributeResult] = []
        for backend in self._backends:
            try:
                results.append(backend.predict(audio, sample_rate, text=text))
            except Exception:
                continue
        if not results:
            return AttributeResult(
                axis=self._axis,
                label="",
                confidence=0.0,
                backend="ensemble:none",
            )
        if self._strategy == "first" or len(results) == 1:
            return results[0]
        if self._strategy == "vote":
            return self._vote(results)
        return self._mean_posterior(results)

    def _vote(self, results: List[AttributeResult]) -> AttributeResult:
        labels = [r.label for r in results if r.label]
        if not labels:
            return AttributeResult(axis=self._axis, label="", backend="ensemble:vote")
        counter = Counter(labels)
        top_label, top_count = counter.most_common(1)[0]
        winners = [r for r in results if r.label == top_label]
        avg_conf = sum(r.confidence for r in winners) / len(winners)
        agreement = top_count / len(results)
        return AttributeResult(
            axis=self._axis,
            label=top_label,
            confidence=avg_conf * agreement,
            backend="ensemble:vote",
            raw={
                "votes": dict(counter),
                "agreement": agreement,
                "winner_backends": [r.backend for r in winners],
            },
        )

    def _mean_posterior(self, results: List[AttributeResult]) -> AttributeResult:
        with_post = [r for r in results if r.posterior]
        if not with_post:
            return self._vote(results)
        agg: Dict[str, float] = {}
        for r in with_post:
            for label, prob in r.posterior.items():
                agg[label] = agg.get(label, 0.0) + prob
        normalized = {k: v / len(with_post) for k, v in agg.items()}
        top_label = max(normalized, key=normalized.get)
        return AttributeResult(
            axis=self._axis,
            label=top_label,
            confidence=normalized[top_label],
            posterior=normalized,
            backend="ensemble:mean_posterior",
            raw={
                "contributing_backends": [r.backend for r in with_post],
            },
        )
