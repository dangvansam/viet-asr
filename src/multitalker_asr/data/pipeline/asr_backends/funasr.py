import re
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

from .base import ASRBackendError, ASRResult, BaseASRBackend


_EMOTION_TAG_MAP: Dict[str, str] = {
    "HAPPY": "happy",
    "SAD": "sad",
    "ANGRY": "angry",
    "NEUTRAL": "neutral",
    "FEARFUL": "fear",
    "SURPRISED": "surprise",
    "DISGUSTED": "disgust",
}

_LANGUAGE_TAG_MAP: Dict[str, str] = {
    "vi": "vi",
    "zh": "zh",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "fr": "fr",
    "de": "de",
    "es": "es",
}

# Fun-ASR-MLT-Nano takes language by its Chinese name (not ISO codes like SenseVoice).
_FUNASR_MLT_LANG: Dict[str, str] = {
    "vi": "越南语", "zh": "中文", "en": "英文", "yue": "粤语", "ja": "日文",
    "ko": "韩文", "id": "印尼语", "th": "泰语", "ms": "马来语", "tl": "菲律宾语",
    "ar": "阿拉伯语", "hi": "印地语", "pt": "葡萄牙语", "nl": "荷兰语",
    "pl": "波兰语", "cs": "捷克语", "da": "丹麦语", "fi": "芬兰语", "el": "希腊语",
    "hu": "匈牙利语", "ro": "罗马尼亚语", "sv": "瑞典语",
}


class FunASRBackend(BaseASRBackend):
    name = "funasr"
    languages = ["vi", "en", "zh", "ja", "ko", "fr", "de", "es"]

    def __init__(
        self,
        model_id: str = "iic/SenseVoiceSmall",
        default_language: str = "vi",
        use_itn: bool = True,
        batch_size_s: int = 60,
    ):
        self._model_id = model_id
        self._default_language = default_language
        self._use_itn = use_itn
        self._batch_size_s = batch_size_s
        self._model = None
        self._loaded = False
        self._device = "cpu"
        # Fun-ASR(-MLT)-Nano has a different generate API (list input, Chinese
        # language names, `itn`/`batch_size`) than SenseVoice. Detect by model id/path.
        self._is_fun_asr = "fun-asr" in model_id.lower()

    def load(self, device: str = "cpu") -> None:
        if self._loaded:
            return
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise ASRBackendError(
                "funasr is required for FunASRBackend. Install via `uv add funasr`."
            ) from exc

        logger.info(f"Loading FunASR model '{self._model_id}' on {device}")
        self._model = AutoModel(model=self._model_id, device=device, disable_update=True)
        self._device = device
        self._loaded = True

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> ASRResult:
        if not self._loaded:
            raise ASRBackendError("FunASRBackend not loaded. Call load() first.")

        lang = language or self._default_language
        if self._is_fun_asr:
            # Fun-ASR-Nano builds its ChatML input from a wav *path* (its frontend
            # doesn't accept raw numpy) → dump to a temp wav and pass the path.
            import tempfile

            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, np.asarray(audio, dtype=np.float32), sample_rate)
                result = self._model.generate(
                    input=[tmp.name],
                    cache={},
                    batch_size=1,
                    language=_FUNASR_MLT_LANG.get(lang, lang),
                    itn=self._use_itn,
                )
        else:
            result = self._model.generate(
                input=audio,
                cache={},
                language=lang,
                use_itn=self._use_itn,
                batch_size_s=self._batch_size_s,
            )
        item = result[0] if result else {}
        raw_text = item.get("text", "")
        return self._parse_result(raw_text, item, lang)

    def _parse_result(
        self, raw_text: str, raw_item: Dict[str, Any], language: str
    ) -> ASRResult:
        text, detected_lang, emotion = self._strip_tags(raw_text)
        return ASRResult(
            text=text,
            confidence=float(raw_item.get("confidence", 1.0)),
            language=detected_lang or language,
            backend=self.name,
            raw={
                "raw_text": raw_text,
                "emotion": emotion,
                "item": raw_item,
            },
        )

    def _strip_tags(self, text: str):
        emotion = None
        language = None
        for match in re.findall(r"<\|([A-Z_a-z]+)\|>", text):
            upper = match.upper()
            if upper in _EMOTION_TAG_MAP:
                emotion = _EMOTION_TAG_MAP[upper]
            elif match.lower() in _LANGUAGE_TAG_MAP:
                language = _LANGUAGE_TAG_MAP[match.lower()]
        clean = re.sub(r"<\|[^|]+\|>", "", text)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean, language, emotion

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            logger.info("FunASRBackend unloaded")
