"""
Qwen3ServiceAlignBackend: HTTP client for a Qwen3-ForcedAligner served in a
SEPARATE venv (qwen-asr downgrades transformers and breaks NeMo in-process, so
serving stays out of the project venv — see scripts/serve_qwen3_aligner.py).

In-venv this backend is pure `requests`: POST the segment wav + text to /align and
parse the returned word spans. Best Vietnamese alignment quality (empirically),
without the transformers/NeMo conflict.
"""

from typing import List

from loguru import logger

from .base import AlignBackendError, AlignedWord, AlignResult, BaseAlignBackend, alignment_score


class Qwen3ServiceAlignBackend(BaseAlignBackend):
    name = "qwen3_service"
    languages: List[str] = []

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8103",
        language: str = "Vietnamese",
        timeout: float = 120.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._language = language
        self._timeout = timeout
        self._session = None
        self._loaded = False

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        try:
            import requests
        except ImportError as exc:
            raise AlignBackendError("requests is required for Qwen3ServiceAlignBackend.") from exc
        self._session = requests.Session()
        self._loaded = True
        logger.info(f"Qwen3ServiceAlignBackend ready (base_url={self._base_url})")

    def align(self, audio_path: str, text: str, language: str) -> AlignResult:
        if not self._loaded:
            raise AlignBackendError("Qwen3ServiceAlignBackend not loaded. Call load() first.")
        if not text.strip():
            return AlignResult(words=[], score=0.0, backend=self.name)

        lang = language or self._language
        try:
            with open(audio_path, "rb") as f:
                response = self._session.post(
                    f"{self._base_url}/v1/audio/alignments",
                    files={"file": ("audio.wav", f, "audio/wav")},
                    data={"text": text, "language": lang},
                    timeout=self._timeout,
                )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise AlignBackendError(f"Qwen3ServiceAlignBackend request failed: {exc}") from exc

        words = [
            AlignedWord(
                text=str(w.get("word", w.get("text", ""))),
                start_time=float(w.get("start", w.get("start_time", 0.0))),
                end_time=float(w.get("end", w.get("end_time", 0.0))),
            )
            for w in (payload.get("words") or [])
            if w.get("word") or w.get("text")
        ]
        return AlignResult(words=words, score=alignment_score(words), backend=self.name)

    def unload(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        self._loaded = False
