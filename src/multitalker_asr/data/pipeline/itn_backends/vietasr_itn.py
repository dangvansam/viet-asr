from pathlib import Path
from typing import Optional

from loguru import logger

from .base import BaseITNBackend, ITNBackendError, ITNResult


class VietASRITN(BaseITNBackend):
    """FST-based Vietnamese ITN from /home/samdv/vietasr/scripts/gen_itn_fst.py."""

    name = "vietasr_itn"
    languages = ["vi"]

    def __init__(
        self,
        fst_path: Optional[str] = None,
        binding_module: str = "vietasr",
    ):
        self._fst_path = Path(fst_path) if fst_path else None
        self._binding_module = binding_module
        self._normalizer = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        try:
            module = __import__(self._binding_module)
        except ImportError as exc:
            raise ITNBackendError(
                f"VietASR binding '{self._binding_module}' not importable."
            ) from exc

        normalizer_cls = self._resolve_normalizer_class(module)
        logger.info(f"Loading VietASR ITN normalizer from {self._fst_path}")
        if self._fst_path is not None:
            self._normalizer = normalizer_cls(fst_path=str(self._fst_path))
        else:
            self._normalizer = normalizer_cls()
        self._loaded = True

    def _resolve_normalizer_class(self, module):
        for attr in ("ITNNormalizer", "Normalizer", "TextNormalizer"):
            cls = getattr(module, attr, None)
            if cls is not None:
                return cls
        raise ITNBackendError(
            f"Could not find ITN normalizer class in '{self._binding_module}'. "
            f"Expected one of: ITNNormalizer, Normalizer, TextNormalizer."
        )

    def normalize(self, text: str, language: str = "vi") -> ITNResult:
        if not self._loaded:
            raise ITNBackendError("VietASRITN not loaded. Call load() first.")
        if not text:
            return ITNResult(text_itn="", backend=self.name, language="vi")
        normalized = self._normalizer.normalize(text)
        return ITNResult(
            text_itn=str(normalized),
            text_spoken=text,
            confidence=1.0,
            backend=self.name,
            language="vi",
        )

    def unload(self) -> None:
        if self._normalizer is not None:
            del self._normalizer
            self._normalizer = None
            self._loaded = False
