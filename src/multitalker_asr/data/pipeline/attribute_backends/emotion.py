from typing import Optional

import numpy as np
from loguru import logger

from .base import AttributeBackendError, AttributeResult, BaseAttributeBackend


_HUBERT_LABEL_MAP = {
    "ang": "angry",
    "hap": "happy",
    "neu": "neutral",
    "sad": "sad",
    "fea": "fear",
    "dis": "disgust",
    "sur": "surprise",
}


class HuBERTEmotionBackend(BaseAttributeBackend):
    axis = "emotion"
    name = "hubert_superb_er"
    label_space = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]

    def __init__(self, model_id: str = "superb/hubert-large-superb-er"):
        self._model_id = model_id
        self._pipeline = None
        self._loaded = False

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise AttributeBackendError(
                "transformers is required for HuBERTEmotionBackend."
            ) from exc
        device_id = self._device_index(device)
        logger.info(f"Loading HuBERT emotion model on {device}")
        self._pipeline = pipeline(
            "audio-classification", model=self._model_id, device=device_id
        )
        self._loaded = True

    def _device_index(self, device: str) -> int:
        if device.startswith("cuda"):
            try:
                return int(device.split(":")[1]) if ":" in device else 0
            except ValueError:
                return 0
        return -1

    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int,
        text: Optional[str] = None,
    ) -> AttributeResult:
        if not self._loaded:
            raise AttributeBackendError("HuBERTEmotionBackend not loaded.")
        out = self._pipeline(
            {"array": audio.astype(np.float32, copy=False), "sampling_rate": sample_rate},
            top_k=len(self.label_space),
        )
        posterior = {}
        for item in out:
            raw_label = str(item.get("label", "")).lower()[:3]
            mapped = _HUBERT_LABEL_MAP.get(raw_label, raw_label)
            posterior[mapped] = float(item.get("score", 0.0))
        if not posterior:
            return AttributeResult(axis=self.axis, label="neutral", backend=self.name)
        top_label = max(posterior, key=posterior.get)
        return AttributeResult(
            axis=self.axis,
            label=top_label,
            confidence=posterior[top_label],
            posterior=posterior,
            backend=self.name,
        )


class FunASREmotionBackend(BaseAttributeBackend):
    axis = "emotion"
    name = "funasr_sensevoice"
    label_space = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]

    def __init__(self, model_id: str = "iic/SenseVoiceSmall"):
        self._model_id = model_id
        self._model = None
        self._loaded = False

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise AttributeBackendError(
                "funasr is required for FunASREmotionBackend."
            ) from exc
        logger.info(f"Loading FunASR SenseVoice model on {device}")
        self._model = AutoModel(model=self._model_id, device=device, disable_update=True)
        self._loaded = True

    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int,
        text: Optional[str] = None,
    ) -> AttributeResult:
        if not self._loaded:
            raise AttributeBackendError("FunASREmotionBackend not loaded.")
        result = self._model.generate(input=audio, cache={}, use_itn=False)
        item = result[0] if result else {}
        raw_text = str(item.get("text", ""))
        label = self._extract_emotion(raw_text)
        return AttributeResult(
            axis=self.axis,
            label=label,
            confidence=1.0 if label != "neutral" else 0.5,
            backend=self.name,
            raw={"raw_text": raw_text},
        )

    def _extract_emotion(self, text: str) -> str:
        import re

        match = re.search(r"<\|([A-Z]+)\|>", text)
        if not match:
            return "neutral"
        token = match.group(1)
        return {
            "HAPPY": "happy",
            "SAD": "sad",
            "ANGRY": "angry",
            "NEUTRAL": "neutral",
            "FEARFUL": "fear",
            "DISGUSTED": "disgust",
            "SURPRISED": "surprise",
        }.get(token, "neutral")
