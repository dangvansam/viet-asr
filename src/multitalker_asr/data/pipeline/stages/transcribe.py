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
        self._client = None
        self._loader = None

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
                        record["text"] = fields["text"]
                        record["text_raw"] = fields.get("text_raw", fields["text"])
                        record["text_itn"] = fields["text_itn"]
                        record["alignment"] = fields.get("alignment", [])
                        record["alignment_score"] = fields.get("alignment_score", 0.0)
                        # Preserve tag-derived emotion/language from the seed.
                        record.setdefault("emotion", fields["emotion"])
                        record.setdefault("language", fields["language"])

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
        """Load the ASR source — a remote service client (base_url) or in-venv FunASR."""
        if getattr(cfg, "base_url", ""):
            if self._client is not None:
                return
            from ....utils.audio import AudioLoader
            from ..asr_backends import build_asr_backend

            self._client = build_asr_backend(
                "openai_transcription",
                base_url=cfg.base_url,
                model=cfg.model or "",
                language=None if cfg.language == "auto" else cfg.language,
                timeout=getattr(cfg, "timeout", 120.0),
            )
            self._client.load()
            self._loader = AudioLoader(target_sample_rate=getattr(cfg, "sample_rate", 16000))
            logger.info(f"TranscribeStage using ASR service {cfg.base_url}")
            return
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
        if self._client is not None:
            return self._transcribe_via_service(audio_path, cfg)
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
            return {"text": "", "text_raw": "", "text_itn": "", "emotion": "neutral",
                    "language": "vi", "alignment": [], "alignment_score": 0.0}

        r = result[0]
        # Fun-ASR-MLT: `text` = normalized (ITN + punctuation), `text_tn` = spoken
        # (un-normalized) form. Keep both; `text` is the primary training text.
        normalized = r.get("text", "")
        raw_spoken = r.get("text_tn") or r.get("raw_text") or normalized
        emotion_raw = r.get("emotion", "")
        language_raw = r.get("language", "")
        alignment = self._build_alignment(r.get("timestamps"))

        return {
            "text": normalized,
            "text_raw": raw_spoken,
            "text_itn": normalized,
            "emotion": _parse_emotion_tag(emotion_raw),
            "language": _parse_language_tag(language_raw),
            "alignment": alignment,
            "alignment_score": self._alignment_score(alignment),
        }

    def _transcribe_via_service(self, audio_path: str, cfg: TranscribeConfig) -> Dict:
        """Transcribe through the unified OpenAI `/v1/audio/transcriptions` client."""
        audio, sr = self._loader.load(audio_path)
        lang = None if cfg.language == "auto" else cfg.language
        result = self._client.transcribe(audio, sr, lang)
        text = result.text or ""
        alignment = [
            {"text": w.word, "start_time": float(w.start), "end_time": float(w.end)}
            for w in (result.word_timings or [])
        ]
        return {
            "text": text,
            "text_raw": text,
            "text_itn": text,
            "emotion": "neutral",
            "language": result.language or "vi",
            "alignment": alignment,
            "alignment_score": self._alignment_score(alignment),
        }

    def _build_alignment(self, timestamps) -> List[Dict]:
        """Convert Fun-ASR-MLT per-token timestamps to alignment records (seconds)."""
        if not timestamps:
            return []
        out: List[Dict] = []
        for t in timestamps:
            token = (t.get("token") or "").strip()
            if not token:
                continue
            out.append({
                "text": token,
                "start_time": float(t.get("start_time", 0.0)),
                "end_time": float(t.get("end_time", 0.0)),
            })
        return out

    def _alignment_score(self, alignment: List[Dict]) -> float:
        if not alignment:
            return 0.0
        valid = sum(
            1 for a in alignment
            if a["start_time"] >= 0 and a["end_time"] > a["start_time"]
        )
        return valid / len(alignment)

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
        """Free GPU memory from FunASR model (no-op for the service client)."""
        if self._client is not None:
            self._client.unload()
            self._client = None
            return
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
