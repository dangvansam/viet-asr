from typing import Optional

from loguru import logger

from .base import BaseITNBackend, ITNBackendError, ITNResult


class NeMoTextNorm(BaseITNBackend):
    """WFST-based ITN via nemo_text_processing (Sparrowhawk-style)."""

    name = "nemo_itn"
    languages = ["en", "es", "de", "fr", "it", "ru", "pt", "vi"]

    def __init__(self, default_language: str = "en", cache_dir: Optional[str] = None):
        self._default_language = default_language
        self._cache_dir = cache_dir
        self._normalizer = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from nemo_text_processing.inverse_text_normalization.inverse_normalize import (
                InverseNormalizer,
            )
        except ImportError as exc:
            raise ITNBackendError(
                "nemo_text_processing is required for NeMoTextNorm. "
                "Install via `pip install nemo_text_processing`."
            ) from exc

        kwargs = {"lang": self._default_language}
        if self._cache_dir is not None:
            kwargs["cache_dir"] = self._cache_dir
        logger.info(f"Loading NeMo InverseNormalizer (lang={self._default_language})")
        self._normalizer = InverseNormalizer(**kwargs)
        self._loaded = True

    def normalize(self, text: str, language: str = "en") -> ITNResult:
        if not self._loaded:
            raise ITNBackendError("NeMoTextNorm not loaded. Call load() first.")
        if not text:
            return ITNResult(text_itn="", backend=self.name, language=language)
        normalized = self._normalizer.normalize(text, verbose=False)
        return ITNResult(
            text_itn=str(normalized),
            text_spoken=text,
            confidence=1.0,
            backend=self.name,
            language=language,
        )

    def unload(self) -> None:
        if self._normalizer is not None:
            del self._normalizer
            self._normalizer = None
            self._loaded = False
