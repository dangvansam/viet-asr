import os
from typing import Any, Dict, List, Optional

from loguru import logger


class FunASRPostProcessor:
    """Wraps Fun-ASR-MLT-Nano for ITN + PnC post-processing.

    Usage:
        pp = FunASRPostProcessor(device="cuda:0")
        pp.load()
        refined = pp.refine_text(audio_path="audio.wav", language="vi", itn=True)
        # → "Xin chào, bạn khỏe không?"

    Reference: /home/samdv/test_asr_punctuation.py
    """

    def __init__(
        self,
        model_name: str = "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
        device: str = "cuda:0",
    ):
        self._model_name = model_name
        self._device = device
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the FunASR model. Requires funasr package."""
        try:
            from funasr import AutoModel
        except ImportError:
            raise ImportError(
                "FunASR is required for ITN/PnC post-processing. "
                "Install with: pip install funasr"
            )

        logger.info(f"Loading FunASR post-processor: {self._model_name}")
        self._model = AutoModel(
            model=self._model_name,
            trust_remote_code=True,
            remote_code="./model.py",
            device=self._device,
        )
        logger.success("FunASR post-processor loaded")

    def refine_text(
        self,
        audio_path: Optional[str] = None,
        raw_text: Optional[str] = None,
        language: str = "auto",
        itn: bool = True,
    ) -> str:
        """Apply ITN + PnC refinement.

        Args:
            audio_path: Path to audio file. If provided, re-transcribes with FunASR.
            raw_text: Raw ASR text (returned as-is if no audio_path).
            language: Language hint ("auto", "vi", "en", etc.)
            itn: Whether to apply text normalization.

        Returns:
            Refined text with ITN and punctuation applied.
        """
        if audio_path is None:
            return raw_text or ""

        if not self.is_loaded:
            self.load()

        if audio_path is not None:
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return raw_text or ""

            try:
                result = self._model.generate(
                    input=[audio_path],
                    cache={},
                    batch_size=1,
                    language=language,
                    itn=itn,
                )
                if result and len(result) > 0:
                    return result[0].get("text", raw_text or "")
                return raw_text or ""
            except Exception as e:
                logger.error(f"FunASR refinement failed: {e}")
                return raw_text or ""

        # No audio path — return raw text as-is
        return raw_text or ""

    def refine_batch(
        self,
        items: List[Dict[str, Any]],
        language: str = "auto",
        itn: bool = True,
    ) -> List[Dict[str, Any]]:
        """Batch refinement. Adds 'refined_text' key to each item.

        Args:
            items: [{"text": str, "audio_path": str, ...}, ...]
            language: Language hint
            itn: Whether to apply ITN

        Returns:
            Items with added 'refined_text' key
        """
        for item in items:
            audio_path = item.get("audio_path")
            raw_text = item.get("text", "")
            item["refined_text"] = self.refine_text(
                audio_path=audio_path,
                raw_text=raw_text,
                language=language,
                itn=itn,
            )
        return items
