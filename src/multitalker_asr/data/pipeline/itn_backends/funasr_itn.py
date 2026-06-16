from typing import Optional

from loguru import logger

from .base import BaseITNBackend, ITNBackendError, ITNResult


class FunASRITN(BaseITNBackend):
    """Use FunASR's MLT-Nano SLU model purely for inverse text normalization."""

    name = "funasr_itn"
    languages = ["vi", "en", "zh", "ja", "ko", "fr", "de", "es"]

    def __init__(
        self,
        model_id: str = "iic/SenseVoiceSmall",
        default_language: str = "vi",
    ):
        self._model_id = model_id
        self._default_language = default_language
        self._model = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise ITNBackendError(
                "funasr is required for FunASRITN. Install via `uv add funasr`."
            ) from exc
        logger.info(f"Loading FunASR ITN model '{self._model_id}'")
        self._model = AutoModel(model=self._model_id, device="cpu", disable_update=True)
        self._loaded = True

    def normalize(self, text: str, language: str = "vi") -> ITNResult:
        if not self._loaded:
            raise ITNBackendError("FunASRITN not loaded. Call load() first.")
        if not text:
            return ITNResult(text_itn="", backend=self.name, language=language)

        normalized = self._normalize_via_model(text, language)
        return ITNResult(
            text_itn=normalized,
            text_spoken=text,
            confidence=1.0,
            backend=self.name,
            language=language,
        )

    def _normalize_via_model(self, text: str, language: str) -> str:
        try:
            output = self._model.normalize(text=text, language=language)
        except AttributeError:
            try:
                output = self._model.tn(text=text, language=language)
            except AttributeError as exc:
                raise ITNBackendError(
                    "FunASR model exposes neither normalize() nor tn()"
                ) from exc
        if isinstance(output, dict):
            return str(output.get("text", text))
        if isinstance(output, list) and output:
            first = output[0]
            if isinstance(first, dict):
                return str(first.get("text", text))
            return str(first)
        return str(output)

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
