import re
from typing import List

from loguru import logger

from .base import AlignBackendError, AlignedWord, AlignResult, BaseAlignBackend, alignment_score


class MMSAlignBackend(BaseAlignBackend):
    name = "mms_fa"
    languages: List[str] = []

    def __init__(self, device: str = "cpu"):
        self._device = device
        self._model = None
        self._tokenizer = None
        self._aligner = None
        self._loaded = False

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        try:
            import torchaudio
        except ImportError as exc:
            raise AlignBackendError(
                "torchaudio is required for MMSAlignBackend."
            ) from exc

        import torch

        self._device = device
        bundle = torchaudio.pipelines.MMS_FA
        logger.info("Loading torchaudio MMS_FA aligner")
        self._model = bundle.get_model().to(torch.device(device))
        self._tokenizer = bundle.get_tokenizer()
        self._aligner = bundle.get_aligner()
        self._sample_rate = bundle.sample_rate
        self._vocab = set(bundle.get_dict())          # 29 romanized chars (a-z + ' -)
        self._loaded = True

    def align(self, audio_path: str, text: str, language: str) -> AlignResult:
        if not self._loaded:
            raise AlignBackendError("MMSAlignBackend not loaded. Call load() first.")
        try:
            emission, ratio = self._compute_emission(audio_path)
        except Exception as exc:
            logger.error(f"MMS_FA emission error for {audio_path}: {exc}")
            return AlignResult(words=[], score=0.0, backend=self.name)
        return self._align_text(emission, ratio, text)

    def align_batch(self, items: List[tuple]) -> List[AlignResult]:
        """items: List[(audio_path, text, language)]. Groups by audio so the
        expensive wav2vec2 emission is computed once per clip and reused across all
        its texts (transcript_refine force-aligns ~6 hypotheses per segment)."""
        if not self._loaded:
            raise AlignBackendError("MMSAlignBackend not loaded. Call load() first.")
        results: List[AlignResult] = [None] * len(items)
        groups: dict = {}
        for i, item in enumerate(items):
            groups.setdefault(item[0], []).append(i)
        for audio_path, idxs in groups.items():
            try:
                emission, ratio = self._compute_emission(audio_path)
            except Exception as exc:
                logger.error(f"MMS_FA emission error for {audio_path}: {exc}")
                for i in idxs:
                    results[i] = AlignResult(words=[], score=0.0, backend=self.name)
                continue
            for i in idxs:
                results[i] = self._align_text(emission, ratio, items[i][1])
        return results

    def _compute_emission(self, audio_path: str):
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio

        # Load via soundfile, not torchaudio.load (torchaudio 2.10 routes through
        # torchcodec, which isn't installed — same reason vad_diarize uses sf).
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        waveform = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0)  # [1, N]
        if sr != self._sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self._sample_rate)
        with torch.inference_mode():
            emission, _ = self._model(waveform.to(torch.device(self._device)))
        ratio = waveform.shape[1] / emission.shape[1] / self._sample_rate
        return emission, ratio

    def _align_text(self, emission, ratio: float, text: str) -> AlignResult:
        import torch

        # MMS_FA tokenizes ROMANIZED text (uroman). Romanize for tokenization
        # (đ→d, à→a) but keep the original word (diacritics) for the output.
        pairs = self._prep(text)
        if not pairs:
            return AlignResult(words=[], score=0.0, backend=self.name)
        originals = [orig for orig, _ in pairs]
        romanized = [rom for _, rom in pairs]
        try:
            with torch.inference_mode():
                token_spans = self._aligner(emission[0], self._tokenizer(romanized))
            aligned: List[AlignedWord] = []
            for word, spans in zip(originals, token_spans):
                if not spans:
                    continue
                start = spans[0].start * ratio
                end = spans[-1].end * ratio
                conf = sum(s.score for s in spans) / len(spans)
                aligned.append(AlignedWord(word, float(start), float(end), float(conf)))
        except Exception as exc:
            logger.error(f"MMS_FA forced-align error: {exc}")
            return AlignResult(words=[], score=0.0, backend=self.name)
        return AlignResult(words=aligned, score=alignment_score(aligned), backend=self.name)

    def _prep(self, text: str) -> List[tuple]:
        """(original_word, romanized_token) pairs; drop words with no in-vocab chars."""
        vocab = getattr(self, "_vocab", set("abcdefghijklmnopqrstuvwxyz'-"))
        pairs = []
        for raw in text.split():
            stripped = re.sub(r"^[^\w]+|[^\w]+$", "", raw, flags=re.UNICODE) or raw
            rom = "".join(c for c in self._romanize(stripped).lower() if c in vocab)
            if rom:
                pairs.append((stripped, rom))
        return pairs

    @staticmethod
    def _romanize(word: str) -> str:
        try:
            from unidecode import unidecode
            return unidecode(word)
        except Exception:
            import unicodedata
            return unicodedata.normalize("NFKD", word).encode("ascii", "ignore").decode("ascii")

    def unload(self) -> None:
        for attr in ("_model", "_tokenizer", "_aligner"):
            if getattr(self, attr, None) is not None:
                setattr(self, attr, None)
        self._loaded = False
