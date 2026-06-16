import io
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .base import ASRBackendError, ASRResult, BaseASRBackend, WordTiming


class OpenAITranscriptionBackend(BaseASRBackend):
    """Unified ASR client for any OpenAI-compatible `/v1/audio/transcriptions` server.

    One client for every local engine — Qwen3-ASR (vLLM), Fun-ASR-MLT (vLLM), and
    nemotron/vietasr wrapped by the LitServe `TranscriptionLitAPI`. Swap backends by URL.
    """

    name = "openai_transcription"
    languages: List[str] = []

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8101",
        model: str = "",
        language: Optional[str] = None,
        api_key: str = "EMPTY",
        prompt: str = "",
        diarization: bool = False,
        timestamp_granularities: Tuple[str, ...] = ("segment", "word"),
        include: Tuple[str, ...] = ("logprobs",),
        timeout: float = 120.0,
        n_best: int = 1,
        nbest_temperature: float = 0.3,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._language = language
        self._api_key = api_key
        self._prompt = prompt
        self._diarization = diarization
        self._granularities = list(timestamp_granularities)
        self._include = list(include)
        self._timeout = timeout
        self._n_best = int(n_best)
        self._nbest_temperature = float(nbest_temperature)
        self._session = None
        self._loaded = False

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        try:
            import requests
        except ImportError as exc:
            raise ASRBackendError(
                "requests is required for OpenAITranscriptionBackend."
            ) from exc
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._loaded = True
        logger.info(
            f"OpenAITranscriptionBackend ready (base_url={self._base_url} model={self._model})"
        )

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> ASRResult:
        if not self._loaded:
            raise ASRBackendError(
                "OpenAITranscriptionBackend not loaded. Call load() first."
            )

        wav_bytes = self._to_wav_bytes(audio, sample_rate)
        lang = language or self._language

        payload = self._post(wav_bytes, lang, "verbose_json")
        if payload is None:
            # Some engines (e.g. vLLM Qwen3-ASR) don't support verbose_json → no
            # word timestamps available; fall back to plain json (text only).
            payload = self._post(wav_bytes, lang, "json")
        if payload is None:
            raise ASRBackendError("OpenAITranscriptionBackend: server rejected request")
        return self._parse_verbose_json(payload, lang)

    def _post(self, wav_bytes: bytes, lang: Optional[str], response_format: str):
        """POST once; return parsed JSON, or None on a 400 (caller may retry)."""
        data: List[Tuple[str, str]] = [("response_format", response_format)]
        if self._model:
            data.append(("model", self._model))
        if lang:
            data.append(("language", lang))
        if self._prompt:
            data.append(("prompt", self._prompt))
        if self._n_best > 1:
            data.append(("n_best", str(self._n_best)))
            data.append(("nbest_temperature", str(self._nbest_temperature)))
        if response_format == "verbose_json":
            if self._diarization:
                data.append(("diarization", "true"))
            for g in self._granularities:
                data.append(("timestamp_granularities[]", g))
            for inc in self._include:
                data.append(("include[]", inc))
        try:
            response = self._session.post(
                f"{self._base_url}/v1/audio/transcriptions",
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data=data,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
        except Exception as exc:
            raise ASRBackendError(
                f"OpenAITranscriptionBackend request failed: {exc}"
            ) from exc
        if response.status_code == 400 and response_format == "verbose_json":
            return None
        try:
            response.raise_for_status()
        except Exception as exc:
            raise ASRBackendError(
                f"OpenAITranscriptionBackend request failed: {exc}"
            ) from exc
        return response.json()

    def _parse_verbose_json(self, payload: dict, language: Optional[str]) -> ASRResult:
        text = (payload.get("text") or "").strip()
        words = self._collect_words(payload)
        confidence = self._confidence(payload, words)
        candidates = self._collect_candidates(payload, text)
        raw = dict(payload) if isinstance(payload, dict) else {}
        if len(candidates) > 1:
            raw["candidates"] = candidates
        else:
            raw.pop("candidates", None)
        return ASRResult(
            text=text,
            confidence=confidence,
            language=payload.get("language") or language or "vi",
            word_timings=self._to_word_timings(words) or None,
            backend=self.name,
            raw=raw,
        )

    @staticmethod
    def _collect_candidates(payload: dict, text: str) -> List[str]:
        raw = payload.get("candidates") or []
        candidates: List[str] = []
        for cand in raw:
            value = (cand.get("text") if isinstance(cand, dict) else cand) or ""
            value = value.strip()
            if value:
                candidates.append(value)
        if candidates and candidates[0] != text and text:
            candidates = [text] + [c for c in candidates if c != text]
        return candidates

    @staticmethod
    def _collect_words(payload: dict) -> List[Dict[str, Any]]:
        if payload.get("words"):
            return payload["words"]
        words: List[Dict[str, Any]] = []
        for segment in payload.get("segments") or []:
            for w in segment.get("words") or []:
                if "speaker" not in w and segment.get("speaker") is not None:
                    w = {**w, "speaker": segment["speaker"]}
                words.append(w)
        return words

    @staticmethod
    def _to_word_timings(words: List[Dict[str, Any]]) -> List[WordTiming]:
        timings: List[WordTiming] = []
        for w in words:
            logprob = w.get("logprob")
            timings.append(
                WordTiming(
                    word=w.get("word", ""),
                    start=float(w.get("start", 0.0)),
                    end=float(w.get("end", 0.0)),
                    confidence=math.exp(logprob) if logprob is not None else None,
                )
            )
        return timings

    @staticmethod
    def _confidence(payload: dict, words: List[Dict[str, Any]]) -> float:
        logprobs = [w["logprob"] for w in words if w.get("logprob") is not None]
        if not logprobs:
            seg_lp = [
                s["avg_logprob"]
                for s in payload.get("segments") or []
                if s.get("avg_logprob") is not None
            ]
            logprobs = seg_lp
        if logprobs:
            return math.exp(sum(logprobs) / len(logprobs))
        return 1.0

    def _to_wav_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise ASRBackendError(
                "soundfile is required for OpenAITranscriptionBackend."
            ) from exc

        buf = io.BytesIO()
        sf.write(buf, audio.astype(np.float32, copy=False), sample_rate, format="WAV")
        return buf.getvalue()

    def unload(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
            self._loaded = False
            logger.info("OpenAITranscriptionBackend unloaded")
