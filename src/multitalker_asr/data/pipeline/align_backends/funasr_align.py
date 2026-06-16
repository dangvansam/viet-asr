"""
FunASRAlignBackend: forced alignment via FunASR's MonotonicAligner (timestamp
prediction model `fa-zh` / `iic/speech_timestamp_prediction-v1-16k-offline`).

Takes (audio, given text) and returns per-token timestamps — same contract as the
other aligners. NOTE: the public model is Mandarin-trained; Vietnamese quality is
the open question this backend exists to evaluate (compare via benchmark_aligners).
"""

from typing import List, Optional, Tuple

from loguru import logger

from .base import AlignBackendError, AlignedWord, AlignResult, BaseAlignBackend, alignment_score


class FunASRAlignBackend(BaseAlignBackend):
    name = "funasr_align"
    languages: List[str] = []          # algorithm is language-agnostic; weights are Mandarin

    def __init__(
        self,
        model: str = "iic/speech_timestamp_prediction-v1-16k-offline",
        device: str = "cuda",
        disable_update: bool = True,
    ):
        self._model_id = model
        self._device = device
        self._disable_update = disable_update
        self._model = None
        self._loaded = False

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise AlignBackendError(
                "funasr is required for FunASRAlignBackend. Install via `uv add funasr`."
            ) from exc
        logger.info(f"Loading FunASR MonotonicAligner: {self._model_id}")
        self._model = AutoModel(
            model=self._model_id,
            device=device or self._device,
            disable_update=self._disable_update,
        )
        self._loaded = True

    def align(self, audio_path: str, text: str, language: str) -> AlignResult:
        if not self._loaded:
            raise AlignBackendError("FunASRAlignBackend not loaded. Call load() first.")
        if not text.strip():
            return AlignResult(words=[], score=0.0, backend=self.name)

        try:
            res = self._model.generate(
                input=(audio_path, text),
                data_type=("sound", "text"),
                batch_size=1,
            )
        except Exception as exc:
            logger.error(f"FunASR alignment error for {audio_path}: {exc}")
            return AlignResult(words=[], score=0.0, backend=self.name)

        if not res:
            return AlignResult(words=[], score=0.0, backend=self.name)
        spans = self._spans_ms(res[0].get("timestamp") or [])
        words = self._group_to_words(text, spans)
        return AlignResult(words=words, score=alignment_score(words), backend=self.name,
                           raw={"timestamp": res[0].get("timestamp")})

    @staticmethod
    def _spans_ms(timestamp) -> List[Tuple[float, float]]:
        """[[start_ms, end_ms], ...] → [(start_s, end_s), ...]."""
        out: List[Tuple[float, float]] = []
        for pair in timestamp:
            if pair is None or len(pair) < 2:
                continue
            out.append((float(pair[0]) / 1000.0, float(pair[1]) / 1000.0))
        return out

    @staticmethod
    def _group_to_words(text: str, spans: List[Tuple[float, float]]) -> List[AlignedWord]:
        """Map per-token spans to the input's whitespace words. 1:1 when counts
        match; otherwise distribute tokens across words by proportion."""
        words = text.split()
        if not words or not spans:
            return []
        if len(spans) == len(words):
            return [AlignedWord(w, s, e) for w, (s, e) in zip(words, spans)]
        # Distribute spans across words proportionally (FunASR token count differs).
        per = len(spans) / len(words)
        out: List[AlignedWord] = []
        for i, w in enumerate(words):
            lo = int(round(i * per))
            hi = max(lo + 1, int(round((i + 1) * per)))
            chunk = spans[lo:hi] or [spans[min(lo, len(spans) - 1)]]
            out.append(AlignedWord(w, chunk[0][0], chunk[-1][1]))
        return out

    def unload(self) -> None:
        self._model = None
        self._loaded = False
