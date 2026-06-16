from typing import List, Optional

import numpy as np
from loguru import logger

from .base import ASRBackendError, ASRResult, BaseASRBackend


class NemotronStreamingASR(BaseASRBackend):
    name = "nemotron"
    languages = [
        "vi-VN",
        "en-US",
        "en-GB",
        "es-US",
        "es-ES",
        "fr-FR",
        "fr-CA",
        "it-IT",
        "pt-BR",
        "pt-PT",
        "nl-NL",
        "de-DE",
        "tr-TR",
        "ru-RU",
        "ar-AR",
        "hi-IN",
        "ja-JP",
        "ko-KR",
        "uk-UA",
    ]

    def __init__(
        self,
        model_name: str = "nvidia/nemotron-3.5-asr-streaming-0.6b",
        target_lang: str = "vi-VN",
        att_context_size: Optional[List[int]] = None,
        strip_lang_tags: bool = True,
    ):
        self._model_name = model_name
        self._target_lang = target_lang
        self._att_context_size = att_context_size or [56, 13]
        self._strip_lang_tags = strip_lang_tags
        self._model = None
        self._loaded = False
        self._device = "cpu"

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise ASRBackendError(
                "nemo-toolkit is required for NemotronStreamingASR."
            ) from exc

        logger.info(f"Loading Nemotron ASR '{self._model_name}' on {device}")
        self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=self._model_name)
        self._model = self._model.to(device)
        self._model.eval()
        encoder = getattr(self._model, "encoder", None)
        if encoder is not None and hasattr(encoder, "set_default_att_context_size"):
            encoder.set_default_att_context_size(self._att_context_size)
        self._device = device
        self._loaded = True

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> ASRResult:
        if not self._loaded:
            raise ASRBackendError("NemotronStreamingASR not loaded. Call load() first.")

        target_lang = self._normalize_lang(language) or self._target_lang
        audio_resampled = self._ensure_16k(audio, sample_rate)

        hyps = self._model.transcribe(
            audio=[audio_resampled],
            batch_size=1,
            target_lang=target_lang,
        )
        return self._build_result(self._extract_text(hyps), target_lang)

    def transcribe_batch(
        self,
        audios: List[np.ndarray],
        sample_rate: int,
        language: Optional[str] = None,
    ) -> List[ASRResult]:
        if not self._loaded:
            raise ASRBackendError("NemotronStreamingASR not loaded. Call load() first.")
        if not audios:
            return []
        target_lang = self._normalize_lang(language) or self._target_lang
        resampled = [self._ensure_16k(a, sample_rate) for a in audios]
        hyps = self._model.transcribe(
            audio=resampled,
            batch_size=len(resampled),
            target_lang=target_lang,
        )
        return [self._build_result(self._hyp_text(h), target_lang)
                for h in self._as_hyp_list(hyps, len(resampled))]

    def _build_result(self, text_raw: str, target_lang: str) -> ASRResult:
        text, detected_lang = self._parse_lang_tag(text_raw)
        return ASRResult(
            text=text if self._strip_lang_tags else text_raw,
            confidence=1.0,
            language=detected_lang or target_lang,
            backend=self.name,
            raw={"raw_text": text_raw, "target_lang": target_lang},
        )

    def _hyp_text(self, hyp) -> str:
        if hasattr(hyp, "text"):
            return hyp.text or ""
        if isinstance(hyp, str):
            return hyp
        return ""

    def _as_hyp_list(self, hyps, n: int) -> list:
        if isinstance(hyps, tuple):
            hyps = hyps[0]
        return list(hyps)[:n]

    def _normalize_lang(self, language):
        """Map ISO codes / names (e.g. 'vi', 'Vietnamese') to nemotron's locale
        tags ('vi-VN'). Unknown values return None so we fall back to target_lang
        instead of erroring with 'Unknown target language'."""
        if not language:
            return None
        if language in self.languages:
            return language
        short = {
            "vi": "vi-VN", "vietnamese": "vi-VN", "en": "en-US", "english": "en-US",
            "zh": "zh-CN", "chinese": "zh-CN", "ja": "ja-JP", "japanese": "ja-JP",
            "ko": "ko-KR", "korean": "ko-KR", "fr": "fr-FR", "de": "de-DE",
            "es": "es-ES", "pt": "pt-BR", "ru": "ru-RU", "ar": "ar-AR", "hi": "hi-IN",
        }
        return short.get(language.strip().lower())

    def _ensure_16k(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate == 16000:
            return audio.astype(np.float32, copy=False)
        try:
            import librosa
        except ImportError as exc:
            raise ASRBackendError(
                "librosa is required to resample audio for Nemotron."
            ) from exc
        return librosa.resample(
            audio.astype(np.float32, copy=False),
            orig_sr=sample_rate,
            target_sr=16000,
        )

    def _extract_text(self, hyps) -> str:
        if not hyps:
            return ""
        first = hyps[0]
        if hasattr(first, "text"):
            return first.text or ""
        if isinstance(first, str):
            return first
        return ""

    def _parse_lang_tag(self, text: str):
        import re

        match = re.search(r"<([a-z]{2}-[A-Z]{2})>", text)
        if match:
            detected = match.group(1)
            clean = text[: match.start()] + text[match.end() :]
            return clean.strip(), detected
        return text, None

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            logger.info("NemotronStreamingASR unloaded")
