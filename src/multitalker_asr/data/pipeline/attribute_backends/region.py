from typing import Dict, Optional

import numpy as np
from loguru import logger

from .base import AttributeBackendError, AttributeResult, BaseAttributeBackend


class FormantHeuristicRegionBackend(BaseAttributeBackend):
    """Coarse rule-based Vietnamese dialect heuristic.

    Uses formant centroid and pitch range as a placeholder until a trained
    classifier is available. Returns conservative confidence.
    """

    axis = "region"
    name = "formant_heuristic"
    label_space = ["northern", "central", "southern"]

    def __init__(
        self,
        northern_centroid_hz: float = 1500.0,
        southern_centroid_hz: float = 1800.0,
    ):
        self._north_centroid = northern_centroid_hz
        self._south_centroid = southern_centroid_hz
        self._loaded = True

    def load(self, device: str = "cpu") -> None:
        self._loaded = True

    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int,
        text: Optional[str] = None,
    ) -> AttributeResult:
        centroid = self._spectral_centroid(audio, sample_rate)
        if centroid is None:
            return AttributeResult(
                axis=self.axis,
                label="northern",
                confidence=0.4,
                backend=self.name,
            )

        distances = {
            "northern": abs(centroid - self._north_centroid),
            "southern": abs(centroid - self._south_centroid),
            "central": abs(centroid - (self._north_centroid + self._south_centroid) / 2),
        }
        label = min(distances, key=distances.get)
        spread = max(distances.values()) - min(distances.values())
        confidence = min(0.7, 0.4 + spread / 1000.0)
        return AttributeResult(
            axis=self.axis,
            label=label,
            confidence=confidence,
            backend=self.name,
            raw={"centroid_hz": centroid, "distances": distances},
        )

    def _spectral_centroid(self, audio: np.ndarray, sample_rate: int):
        try:
            import librosa
        except ImportError:
            return None
        try:
            centroid = librosa.feature.spectral_centroid(
                y=audio.astype(np.float32, copy=False), sr=sample_rate
            )
            return float(centroid.mean())
        except Exception:
            return None


class LoadedRegionClassifier(BaseAttributeBackend):
    """Wrapper for a torch-script Vietnamese dialect classifier."""

    axis = "region"
    name = "torch_dialect_clf"
    label_space = ["northern", "central", "southern"]

    def __init__(
        self,
        checkpoint_path: str,
        sample_rate: int = 16000,
    ):
        self._checkpoint_path = checkpoint_path
        self._sample_rate = sample_rate
        self._model = None
        self._loaded = False
        self._device = "cpu"

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        try:
            import torch
        except ImportError as exc:
            raise AttributeBackendError(
                "torch is required for LoadedRegionClassifier."
            ) from exc
        logger.info(f"Loading region classifier from {self._checkpoint_path}")
        self._model = torch.jit.load(self._checkpoint_path, map_location=device)
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
            raise AttributeBackendError("LoadedRegionClassifier not loaded.")
        if sample_rate != self._sample_rate:
            try:
                import librosa

                audio = librosa.resample(
                    audio.astype(np.float32, copy=False),
                    orig_sr=sample_rate,
                    target_sr=self._sample_rate,
                )
            except ImportError as exc:
                raise AttributeBackendError(
                    "librosa is required to resample for LoadedRegionClassifier."
                ) from exc

        wav = self._torch.from_numpy(audio.astype(np.float32, copy=False)).unsqueeze(0)
        wav = wav.to(self._device)
        with self._torch.no_grad():
            logits = self._model(wav)
        probs = self._torch.softmax(logits[0], dim=-1).cpu().numpy()
        posterior = {
            self.label_space[idx]: float(probs[idx])
            for idx in range(min(len(self.label_space), len(probs)))
        }
        top_label = max(posterior, key=posterior.get)
        return AttributeResult(
            axis=self.axis,
            label=top_label,
            confidence=posterior[top_label],
            posterior=posterior,
            backend=self.name,
        )
