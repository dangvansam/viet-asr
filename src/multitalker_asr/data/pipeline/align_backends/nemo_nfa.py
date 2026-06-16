from typing import List, Optional

from loguru import logger

from .base import AlignBackendError, AlignedWord, AlignResult, BaseAlignBackend, alignment_score

_WORD_MARKER = "▁"


class NeMoNFABackend(BaseAlignBackend):
    name = "nemo_nfa"
    languages: List[str] = []

    def __init__(
        self,
        model: str = "",
        model_path: Optional[str] = None,
    ):
        self._model_name = model
        self._model_path = model_path
        self._model = None
        self._loaded = False

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        try:
            from nemo.collections.asr.models import ASRModel
        except ImportError as exc:
            raise AlignBackendError(
                "nemo-toolkit is required for NeMoNFABackend."
            ) from exc

        if not self._model_name and not self._model_path:
            raise AlignBackendError(
                "NeMoNFABackend requires a CTC `model` name or `model_path`."
            )

        logger.info(f"Loading NeMo CTC model for NFA: {self._model_path or self._model_name}")
        if self._model_path:
            self._model = ASRModel.restore_from(self._model_path, map_location=device)
        else:
            self._model = ASRModel.from_pretrained(self._model_name, map_location=device)
        self._model.eval()
        self._loaded = True

    def align(self, audio_path: str, text: str, language: str) -> AlignResult:
        if not self._loaded:
            raise AlignBackendError("NeMoNFABackend not loaded. Call load() first.")

        try:
            import soundfile as sf
            import torch
            import torchaudio

            log_probs = self._model.transcribe([audio_path], logprobs=True)[0]
            emission = torch.as_tensor(log_probs, dtype=torch.float32).unsqueeze(0)

            token_ids = self._model.tokenizer.text_to_ids(text)
            if not token_ids:
                return AlignResult(words=[], score=0.0, backend=self.name)
            targets = torch.tensor([token_ids], dtype=torch.int32)
            blank = emission.shape[-1] - 1

            aligned, scores = torchaudio.functional.forced_align(
                emission, targets, blank=blank
            )
            token_spans = torchaudio.functional.merge_tokens(aligned[0], scores[0].exp())

            duration = sf.info(audio_path).duration
            ratio = duration / emission.shape[1] if emission.shape[1] > 0 else 0.0
            words = self._group_words(token_spans, ratio)
        except Exception as exc:
            logger.error(f"NeMo NFA alignment error for {audio_path}: {exc}")
            return AlignResult(words=[], score=0.0, backend=self.name)

        return AlignResult(words=words, score=alignment_score(words), backend=self.name)

    def _group_words(self, token_spans, ratio: float) -> List[AlignedWord]:
        words: List[AlignedWord] = []
        pieces: List[str] = []
        start = None
        end = None
        confs: List[float] = []

        def flush():
            if pieces and start is not None:
                text = "".join(pieces).replace(_WORD_MARKER, "")
                if text:
                    conf = sum(confs) / len(confs) if confs else 1.0
                    words.append(AlignedWord(text, start * ratio, end * ratio, float(conf)))

        for span in token_spans:
            piece = self._model.tokenizer.ids_to_tokens([span.token])[0]
            if piece.startswith(_WORD_MARKER) and pieces:
                flush()
                pieces, confs = [], []
                start = None
            if start is None:
                start = span.start
            end = span.end
            pieces.append(piece)
            confs.append(getattr(span, "score", 1.0))
        flush()
        return words

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
