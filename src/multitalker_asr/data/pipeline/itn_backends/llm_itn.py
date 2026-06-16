from typing import Optional

from loguru import logger

from ....configs.llm import LLMConfig
from ....utils.llm_client import LLMClient, LLMClientError
from .base import BaseITNBackend, ITNBackendError, ITNResult


class LLMITNBackend(BaseITNBackend):
    name = "llm_itn"
    languages = ["vi", "en", "zh", "ja", "ko", "fr", "de", "es"]

    DEFAULT_SYSTEM = (
        "You are an expert Vietnamese/multilingual text normalizer. "
        "Restore proper punctuation, capitalization, numbers, and inverse text "
        "normalization. Return ONLY the normalized text, no quotes, no commentary."
    )

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
    ):
        self._llm_config = llm_config or LLMConfig()
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM
        self._user_template = (
            user_template
            or "Language: {language}\nInput: {text}\nNormalized:"
        )
        self._client: Optional[LLMClient] = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._client = LLMClient(self._llm_config)
        self._loaded = True
        logger.info(f"LLMITNBackend ready (model={self._llm_config.model})")

    def normalize(self, text: str, language: str = "vi") -> ITNResult:
        if not self._loaded:
            raise ITNBackendError("LLMITNBackend not loaded. Call load() first.")
        if not text:
            return ITNResult(text_itn="", backend=self.name, language=language)

        user = self._user_template.format(language=language, text=text)
        try:
            response = self._client.complete(self._system_prompt, user)
        except LLMClientError as exc:
            raise ITNBackendError(f"LLM ITN call failed: {exc}") from exc

        normalized = response.text.strip()
        return ITNResult(
            text_itn=normalized,
            text_spoken=text,
            confidence=1.0,
            backend=self.name,
            language=language,
            raw={"usage": response.usage, "model": response.model},
        )

    def unload(self) -> None:
        if self._client is not None:
            self._client = None
            self._loaded = False
