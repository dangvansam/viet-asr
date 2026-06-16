from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from .base import AttributeBackendError, AttributeResult, BaseAttributeBackend


class SpeechBrainLangIDBackend(BaseAttributeBackend):
    axis = "language"
    name = "speechbrain_voxlingua"
    label_space = ["vi-VN", "en-US", "zh-CN", "auto"]

    _ISO_TO_LOCALE = {
        "vi": "vi-VN",
        "en": "en-US",
        "zh": "zh-CN",
        "ja": "ja-JP",
        "ko": "ko-KR",
        "fr": "fr-FR",
        "de": "de-DE",
        "es": "es-ES",
    }

    def __init__(
        self,
        model_id: str = "speechbrain/lang-id-voxlingua107-ecapa",
        savedir: Optional[str] = None,
    ):
        self._model_id = model_id
        self._savedir = savedir
        self._model = None
        self._loaded = False

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        try:
            from speechbrain.inference.classifiers import EncoderClassifier
        except ImportError as exc:
            raise AttributeBackendError(
                "speechbrain is required for SpeechBrainLangIDBackend."
            ) from exc
        run_opts = {"device": device}
        kwargs = {"source": self._model_id, "run_opts": run_opts}
        if self._savedir is not None:
            kwargs["savedir"] = self._savedir
        logger.info(f"Loading SpeechBrain LangID model on {device}")
        self._model = EncoderClassifier.from_hparams(**kwargs)
        self._loaded = True

    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int,
        text: Optional[str] = None,
    ) -> AttributeResult:
        if not self._loaded:
            raise AttributeBackendError("SpeechBrainLangIDBackend not loaded.")
        import torch

        wav = torch.from_numpy(audio.astype(np.float32, copy=False)).unsqueeze(0)
        out = self._model.classify_batch(wav)
        score, _, _, label_text = out
        iso = str(label_text[0]).split(":")[0].strip().lower()[:2]
        locale = self._ISO_TO_LOCALE.get(iso, "auto")
        confidence = float(score.exp().item())
        return AttributeResult(
            axis=self.axis,
            label=locale,
            confidence=confidence,
            backend=self.name,
            raw={"iso": iso, "score": confidence},
        )


class TagDerivedLanguageBackend(BaseAttributeBackend):
    """Use language tag emitted by an upstream ASR backend (e.g. Nemotron auto-detect)."""

    axis = "language"
    name = "tag_derived"
    label_space = ["vi-VN", "en-US", "zh-CN", "auto"]

    def __init__(self, fallback: str = "vi-VN"):
        self._fallback = fallback
        self._loaded = True

    def load(self, device: str = "cpu") -> None:
        self._loaded = True

    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int,
        text: Optional[str] = None,
    ) -> AttributeResult:
        if text:
            for token in text.split():
                token = token.strip("<>")
                if "-" in token and len(token) == 5:
                    return AttributeResult(
                        axis=self.axis,
                        label=token,
                        confidence=0.9,
                        backend=self.name,
                    )
        return AttributeResult(
            axis=self.axis,
            label=self._fallback,
            confidence=0.5,
            backend=self.name,
        )
