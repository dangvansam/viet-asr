import io
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from ....configs.google_speech import GoogleSpeechConfig
from .base import ASRBackendError, ASRResult, BaseASRBackend, WordTiming


class GoogleSpeechBackend(BaseASRBackend):
    """Google Cloud Speech-to-Text V2 backend.

    Supports sync inline recognition (``transcribe``) for short clips driven by the
    numpy pipeline interface, and batch recognition (``transcribe_uri``) for long-form
    audio stored in Cloud Storage. Text normalization is applied via V2
    ``TranscriptNormalization`` and pronunciation hints via ``SpeechAdaptation``.
    """

    name = "google_speech"
    languages = [
        "vi-VN",
        "en-US",
        "en-GB",
        "zh",
        "ja-JP",
        "ko-KR",
        "fr-FR",
        "de-DE",
        "es-ES",
    ]

    def __init__(self, config: Optional[GoogleSpeechConfig] = None, **kwargs):
        if config is None:
            config = GoogleSpeechConfig(**kwargs)
        self._config = config
        self._client = None
        self._types = None
        self._recognizer_path = config.recognizer_path
        self._loaded = False

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        try:
            from google.cloud.speech_v2 import SpeechClient
            from google.cloud.speech_v2.types import cloud_speech
        except ImportError as exc:
            raise ASRBackendError(
                "google-cloud-speech is required for GoogleSpeechBackend. "
                "Install via `uv add google-cloud-speech`."
            ) from exc

        client_kwargs = {}
        if self._config.credentials_path:
            try:
                from google.oauth2 import service_account
            except ImportError as exc:
                raise ASRBackendError(
                    "google-auth is required to use credentials_path."
                ) from exc
            client_kwargs["credentials"] = (
                service_account.Credentials.from_service_account_file(
                    self._config.credentials_path
                )
            )
        if self._config.location != "global":
            client_kwargs["client_options"] = {
                "api_endpoint": f"{self._config.location}-speech.googleapis.com"
            }

        self._types = cloud_speech
        self._client = SpeechClient(**client_kwargs)
        self._loaded = True
        logger.info(
            f"GoogleSpeechBackend ready (project={self._config.project_id} "
            f"model={self._config.model} location={self._config.location})"
        )

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> ASRResult:
        if not self._loaded:
            raise ASRBackendError("GoogleSpeechBackend not loaded. Call load() first.")

        audio_bytes = self._to_wav_bytes(audio, sample_rate)
        config = self._build_recognition_config(language)
        request = self._types.RecognizeRequest(
            recognizer=self._recognizer_path,
            config=config,
            content=audio_bytes,
        )
        response = self._client.recognize(request=request)
        return self._parse_results(response.results, language)

    def transcribe_uri(
        self,
        audio_uri: str,
        language: Optional[str] = None,
    ) -> ASRResult:
        if not self._loaded:
            raise ASRBackendError("GoogleSpeechBackend not loaded. Call load() first.")

        config = self._build_recognition_config(language)
        request = self._types.BatchRecognizeRequest(
            recognizer=self._recognizer_path,
            config=config,
            files=[self._types.BatchRecognizeFileMetadata(uri=audio_uri)],
            recognition_output_config=self._types.RecognitionOutputConfig(
                inline_response_config=self._types.InlineOutputConfig(),
            ),
        )
        operation = self._client.batch_recognize(request=request)
        response = operation.result(timeout=self._config.batch_timeout)
        transcript = response.results[audio_uri].transcript
        return self._parse_results(transcript.results, language)

    def _build_recognition_config(self, language: Optional[str]):
        types = self._types
        language_codes = [language] if language else list(self._config.language_codes)
        word_features = self._config.model != "chirp_3"
        features = types.RecognitionFeatures(
            enable_automatic_punctuation=self._config.enable_automatic_punctuation,
            enable_spoken_punctuation=self._config.enable_spoken_punctuation,
            enable_word_time_offsets=self._config.enable_word_time_offsets
            and word_features,
            enable_word_confidence=self._config.enable_word_confidence
            and word_features,
            profanity_filter=self._config.profanity_filter,
            max_alternatives=self._config.max_alternatives,
        )
        config = types.RecognitionConfig(
            auto_decoding_config=types.AutoDetectDecodingConfig(),
            language_codes=language_codes,
            model=self._config.model,
            features=features,
        )
        adaptation = self._build_adaptation()
        if adaptation is not None:
            config.adaptation = adaptation
        return config

    def _build_adaptation(self):
        if not self._config.phrase_sets:
            return None
        types = self._types
        phrase_sets = []
        for spec in self._config.phrase_sets:
            phrase = types.PhraseSet.Phrase(
                value=spec.get("value", ""),
                boost=spec.get("boost", 0.0),
            )
            phrase_sets.append(
                types.SpeechAdaptation.AdaptationPhraseSet(
                    inline_phrase_set=types.PhraseSet(phrases=[phrase]),
                )
            )
        return types.SpeechAdaptation(phrase_sets=phrase_sets)

    def _parse_results(self, results, language: Optional[str]) -> ASRResult:
        texts: List[str] = []
        confidences: List[float] = []
        word_timings: List[WordTiming] = []
        per_result_alternatives: List[List[Dict[str, Any]]] = []
        for result in results:
            if not result.alternatives:
                continue
            per_result_alternatives.append(
                [
                    {
                        "transcript": alt.transcript.strip(),
                        "confidence": getattr(alt, "confidence", 0.0),
                    }
                    for alt in result.alternatives
                ]
            )
            alternative = result.alternatives[0]
            transcript = alternative.transcript.strip()
            if transcript:
                texts.append(transcript)
            if getattr(alternative, "confidence", 0.0):
                confidences.append(alternative.confidence)
            word_timings.extend(self._parse_word_timings(alternative))

        text = " ".join(texts).strip()
        confidence = sum(confidences) / len(confidences) if confidences else 1.0
        return ASRResult(
            text=text,
            confidence=confidence,
            language=language or self._config.language_codes[0],
            word_timings=word_timings or None,
            backend=self.name,
            raw=self._build_raw(per_result_alternatives),
        )

    @staticmethod
    def _build_raw(
        per_result_alternatives: List[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        has_multiple = any(len(alts) > 1 for alts in per_result_alternatives)
        if not has_multiple:
            return {}
        if len(per_result_alternatives) == 1:
            return {"alternatives": per_result_alternatives[0]}
        return {"alternatives": per_result_alternatives}

    def _parse_word_timings(self, alternative) -> List[WordTiming]:
        timings: List[WordTiming] = []
        for word in getattr(alternative, "words", []) or []:
            timings.append(
                WordTiming(
                    word=word.word,
                    start=self._offset_seconds(getattr(word, "start_offset", None)),
                    end=self._offset_seconds(getattr(word, "end_offset", None)),
                    confidence=getattr(word, "confidence", None) or None,
                )
            )
        return timings

    @staticmethod
    def _offset_seconds(offset: Any) -> float:
        if offset is None:
            return 0.0
        if hasattr(offset, "total_seconds"):
            return offset.total_seconds()
        return float(offset)

    def _to_wav_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise ASRBackendError(
                "soundfile is required for GoogleSpeechBackend."
            ) from exc

        buf = io.BytesIO()
        sf.write(buf, audio.astype(np.float32, copy=False), sample_rate, format="WAV")
        return buf.getvalue()

    def unload(self) -> None:
        if self._client is not None:
            self._client = None
            self._types = None
            self._loaded = False
            logger.info("GoogleSpeechBackend unloaded")
