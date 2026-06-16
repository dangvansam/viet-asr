from typing import Optional

import numpy as np
from loguru import logger

from .base import AttributeBackendError, AttributeResult, BaseAttributeBackend


class Wav2Vec2GenderBackend(BaseAttributeBackend):
    axis = "gender"
    name = "wav2vec2_gender"
    label_space = ["male", "female"]

    def __init__(self, model_id: str = "audeering/wav2vec2-large-robust-24-ft-age-gender"):
        self._model_id = model_id
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
                "transformers is required for Wav2Vec2GenderBackend."
            ) from exc

        logger.info(f"Loading wav2vec2 gender model on {device}")
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
            raise AttributeBackendError("Wav2Vec2GenderBackend not loaded.")

        inputs = self._processor(
            audio.astype(np.float32, copy=False),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            logits = self._model(**inputs).logits[0]
        probs = self._torch.softmax(logits, dim=-1).cpu().numpy()
        posterior = {label: float(probs[idx]) for idx, label in self._iter_id_to_label()}
        if not posterior:
            return AttributeResult(axis=self.axis, label="male", backend=self.name)
        top_label = max(posterior, key=posterior.get)
        return AttributeResult(
            axis=self.axis,
            label=top_label,
            confidence=posterior[top_label],
            posterior=posterior,
            backend=self.name,
        )

    def _iter_id_to_label(self):
        id2label = getattr(self._model.config, "id2label", {})
        for idx_str, label in id2label.items():
            idx = int(idx_str)
            label_low = str(label).lower()
            if "male" in label_low and "fe" not in label_low:
                yield idx, "male"
            elif "female" in label_low:
                yield idx, "female"


class F0HeuristicGenderBackend(BaseAttributeBackend):
    axis = "gender"
    name = "f0_heuristic"
    label_space = ["male", "female"]

    def __init__(self, threshold_hz: float = 165.0):
        self._threshold = threshold_hz
        self._loaded = True

    def load(self, device: str = "cpu") -> None:
        self._loaded = True

    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int,
        text: Optional[str] = None,
    ) -> AttributeResult:
        f0 = self._estimate_f0(audio, sample_rate)
        if f0 is None or f0 <= 0:
            return AttributeResult(
                axis=self.axis,
                label="male",
                confidence=0.5,
                backend=self.name,
                raw={"f0": f0},
            )
        label = "female" if f0 >= self._threshold else "male"
        margin = abs(f0 - self._threshold) / self._threshold
        confidence = min(1.0, 0.5 + margin)
        return AttributeResult(
            axis=self.axis,
            label=label,
            confidence=confidence,
            backend=self.name,
            raw={"f0": f0, "threshold": self._threshold},
        )

    def _estimate_f0(self, audio: np.ndarray, sample_rate: int):
        try:
            import librosa
        except ImportError:
            return None
        try:
            f0, _, _ = librosa.pyin(
                audio.astype(np.float32, copy=False),
                fmin=50.0,
                fmax=400.0,
                sr=sample_rate,
            )
            valid = f0[~np.isnan(f0)]
            if valid.size == 0:
                return None
            return float(np.median(valid))
        except Exception:
            return None
