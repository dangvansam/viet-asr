"""
TranscribeStage: run FunASR MLT-Nano to produce text, ITN text, emotion, language.
"""

import gc
import re
from typing import Dict, List, Optional

from loguru import logger

from ..base_stage import BaseStage
from ..checkpoint import PipelineCheckpoint
from ..config import PipelineConfig, TranscribeConfig

# Emotion tag → standard label mapping
_EMOTION_TAG_MAP: Dict[str, str] = {
    "HAPPY": "happy",
    "SAD": "sad",
    "ANGRY": "angry",
    "NEUTRAL": "neutral",
    "FEARFUL": "fearful",
    "SURPRISED": "surprised",
    "DISGUSTED": "disgusted",
}

# Language tag → ISO code mapping
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


def _parse_emotion_tag(tag: Optional[str]) -> str:
    """Parse FunASR emotion tag like '<|HAPPY|>' → 'happy'. Returns 'neutral' on miss."""
    if not tag:
        return "neutral"
    m = re.search(r"<\|([A-Z]+)\|>", tag)
    if m:
        return _EMOTION_TAG_MAP.get(m.group(1), "neutral")
    return "neutral"


def _parse_language_tag(tag: Optional[str]) -> str:
    """Parse FunASR language tag like '<|vi|>' → 'vi'. Returns 'vi' on miss."""
    if not tag:
        return "vi"
    m = re.search(r"<\|([a-z]+)\|>", tag)
    if m:
        return _LANGUAGE_TAG_MAP.get(m.group(1), m.group(1))
    return "vi"


class TranscribeStage(BaseStage):
    """Transcribe audio segments with FunASR MLT-Nano."""

    name = "transcribe"

    def __init__(self) -> None:
        self._model = None

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        """
        Transcribe audio segments with FunASR MLT-Nano.
        If config.transcribe.itn_only=True: normalize existing text only.
        Frees GPU memory after completing all records.
        """
        to_process, done = self._skip_processed(records, checkpoint)
        if not to_process:
            return done

        cfg = config.transcribe
        self._load_model(cfg)

        processed: List[Dict] = list(done)
        batch_size = cfg.batch_size

        try:
            # Process in batches but call per-file (FunASR batch may OOM)
            for i, record in enumerate(to_process):
                record = dict(record)
                try:
                    if cfg.itn_only:
                        if not record.get("text"):
                            logger.warning(
                                f"itn_only=True but record {record['id']} has no text field"
                            )
                            processed.append(record)
                            continue
                        record["text_itn"] = self._normalize_text_only(record["text"], self._model)
                        # Preserve existing emotion/language if already set
                        if not record.get("emotion"):
                            record["emotion"] = "neutral"
                        if not record.get("language"):
                            record["language"] = "vi"
                    else:
                        audio_path = record.get("audio_filepath", "")
                        from pathlib import Path
                        if not audio_path or not Path(audio_path).exists():
                            logger.error(f"Audio missing for {record['id']}: {audio_path}")
                            record["text"] = ""
                            record["text_itn"] = ""
                            record["emotion"] = "neutral"
                            record["language"] = "vi"
                            processed.append(record)
                            continue
                        fields = self._transcribe_audio(audio_path, self._model, cfg)
                        record.update(fields)

                    checkpoint.mark_processed(record["id"], self.name)

                    # Save checkpoint every batch_size records
                    if (i + 1) % batch_size == 0:
                        checkpoint.save_state()

                except Exception as e:
                    logger.error(f"Transcription failed for {record['id']}: {e}")
                    record.setdefault("text", "")
                    record.setdefault("text_itn", "")
                    record.setdefault("emotion", "neutral")
                    record.setdefault("language", "vi")

                processed.append(record)

            checkpoint.save_state()

        finally:
            self._free_gpu()

        return processed

    def _load_model(self, cfg: TranscribeConfig) -> None:
        """Load FunASR AutoModel. Cached on self._model."""
        if self._model is not None:
            return
        try:
            from funasr import AutoModel
        except ImportError:
            raise ImportError("funasr not installed — run: pip install funasr")

        logger.info(f"Loading FunASR model: {cfg.model}")
        self._model = AutoModel(
            model=cfg.model,
            device=cfg.device,
        )
        logger.info("FunASR model loaded")

    def _transcribe_audio(
        self,
        audio_path: str,
        model,
        cfg: TranscribeConfig,
    ) -> Dict:
        """
        Call model.generate and parse result into standard fields.
        Returns: {"text": str, "text_itn": str, "emotion": str, "language": str}
        """
        try:
            result = model.generate(input=audio_path, language=cfg.language)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM — retrying {audio_path} with batch_size=1")
                import torch
                torch.cuda.empty_cache()
                result = model.generate(input=audio_path, language=cfg.language)
            else:
                raise

        if not result:
            logger.warning(f"Empty FunASR result for {audio_path}")
            return {"text": "", "text_itn": "", "emotion": "neutral", "language": "vi"}

        r = result[0]
        raw_text = r.get("raw_text") or r.get("text", "")
        text_itn = r.get("text", raw_text)
        emotion_raw = r.get("emotion", "")
        language_raw = r.get("language", "")

        return {
            "text": raw_text,
            "text_itn": text_itn,
            "emotion": _parse_emotion_tag(emotion_raw),
            "language": _parse_language_tag(language_raw),
        }

    def _normalize_text_only(self, text: str, model) -> str:
        """Run ITN on existing text string (no audio). Falls back to original text."""
        try:
            result = model.generate(input=text, data_type="text")
            if result:
                return result[0].get("text", text)
        except Exception as e:
            logger.warning(f"ITN text normalization failed: {e}")
        return text

    def _free_gpu(self) -> None:
        """Free GPU memory from FunASR model."""
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
