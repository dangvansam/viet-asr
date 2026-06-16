from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from .base import AttributeBackendError, AttributeResult, BaseAttributeBackend


_DEFAULT_AGE_BINS: List[Tuple[float, str]] = [
    (12.0, "child"),
    (20.0, "teen"),
    (60.0, "adult"),
    (200.0, "senior"),
]


class Wav2Vec2AgeBackend(BaseAttributeBackend):
    axis = "age"
    name = "wav2vec2_age"
    label_space = ["child", "teen", "adult", "senior"]

    def __init__(
        self,
        model_id: str = "audeering/wav2vec2-large-robust-24-ft-age-gender",
        age_bins: Optional[List[Tuple[float, str]]] = None,
    ):
        self._model_id = model_id
        self._age_bins = age_bins or _DEFAULT_AGE_BINS
        self._model = None
        self._processor = None
        self._loaded = False
        self._device = "cpu"

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        except ImportError as exc:
            raise AttributeBackendError(
                "transformers is required for Wav2Vec2AgeBackend."
            ) from exc

        logger.info(f"Loading wav2vec2 age model on {device}")
        self._processor = AutoFeatureExtractor.from_pretrained(self._model_id)
        self._model = AutoModelForAudioClassification.from_pretrained(self._model_id)
        self._model.to(device)
        self._model.eval()
        self._device = device
        self._torch = torch
        self._loaded = True

    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int,
        text: Optional[str] = None,
    ) -> AttributeResult:
        if not self._loaded:
            raise AttributeBackendError("Wav2Vec2AgeBackend not loaded.")

        inputs = self._processor(
            audio.astype(np.float32, copy=False),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            output = self._model(**inputs)
        age_value = self._extract_age_value(output)
        label = self._bin_age(age_value)
        return AttributeResult(
            axis=self.axis,
            label=label,
            confidence=0.7,
            backend=self.name,
            raw={"age_years": age_value},
        )

    def _extract_age_value(self, output) -> float:
        logits = getattr(output, "logits", output)
        if logits.ndim == 2 and logits.shape[-1] == 1:
            return float(logits[0, 0].item()) * 100.0
        if logits.ndim == 2 and logits.shape[-1] > 1:
            probs = self._torch.softmax(logits[0], dim=-1)
            return float((probs * self._torch.arange(probs.shape[0])).sum().item())
        return 30.0

    def _bin_age(self, years: float) -> str:
        for upper, label in self._age_bins:
            if years < upper:
                return label
        return self._age_bins[-1][1]
