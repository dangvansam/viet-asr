from typing import List

from loguru import logger

from .base import AlignBackendError, AlignedWord, AlignResult, BaseAlignBackend, alignment_score


class Qwen3AlignBackend(BaseAlignBackend):
    name = "qwen3"
    languages = [
        "Chinese",
        "English",
        "Cantonese",
        "French",
        "German",
        "Italian",
        "Japanese",
        "Korean",
        "Portuguese",
        "Russian",
        "Spanish",
    ]

    def __init__(
        self,
        model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        dtype: str = "bfloat16",
    ):
        self._model_id = model
        self._dtype = dtype
        self._aligner = None
        self._loaded = False

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        try:
            from qwen_asr import Qwen3ForcedAligner
        except ImportError as exc:
            raise AlignBackendError(
                "qwen_asr is required for Qwen3AlignBackend. Install via `pip install qwen-asr`."
            ) from exc

        import torch

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self._dtype, torch.bfloat16)
        device_map = f"{device}:0" if "cuda" in device else device

        logger.info(f"Loading Qwen3ForcedAligner: {self._model_id}")
        self._aligner = Qwen3ForcedAligner.from_pretrained(
            self._model_id,
            dtype=torch_dtype,
            device_map=device_map,
        )
        self._loaded = True

    def align(self, audio_path: str, text: str, language: str) -> AlignResult:
        if not self._loaded:
            raise AlignBackendError("Qwen3AlignBackend not loaded. Call load() first.")

        try:
            results = self._aligner.align(
                audio=[audio_path],
                text=[text],
                language=[language],
            )
        except Exception as exc:
            logger.error(f"Qwen3 alignment error for {audio_path}: {exc}")
            return AlignResult(words=[], score=0.0, backend=self.name)

        if not results or not results[0]:
            logger.warning(f"Empty alignment for {audio_path}")
            return AlignResult(words=[], score=0.0, backend=self.name)

        words: List[AlignedWord] = [
            AlignedWord(text=item.text, start_time=item.start_time, end_time=item.end_time)
            for item in results[0]
        ]
        return AlignResult(words=words, score=alignment_score(words), backend=self.name)

    def unload(self) -> None:
        if self._aligner is not None:
            del self._aligner
            self._aligner = None
            self._loaded = False
