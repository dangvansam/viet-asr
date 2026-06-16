"""
TranscriptRefineStage: refine the primary transcript (from `transcribe`, Fun-ASR
with punctuation + word timestamps) using additional ASR backends (vietasr,
qwen3_vllm, funasr_mlt) and the crawl VTT subtitle, fused by time-aligned
word-level ROVER.

The primary transcript is the anchor: its words are kept unless out-voted by a
weighted majority of the other backends + the subtitle, so punctuation/casing is
preserved while wrong words get corrected. text_raw / emotion / language from the
primary are left intact; alignment is updated to the fused words.
"""

import gc
from typing import Dict, List, Optional

from loguru import logger

from ....configs.transcript_pipeline import TranscriptPipelineConfig
from ....utils.audio import AudioLoader
from ....utils.subtitle import VTTParser
from ..align_backends import build_align_backend
from ..asr_backends import (
    ASREnsemblerFactory,
    ASRResult,
    BaseASRBackend,
    WordTiming,
    build_asr_backend,
)
from ..base_stage import BaseStage
from ..checkpoint import PipelineCheckpoint
from ..config import PipelineConfig
from ..parallel import parallel_map


class TranscriptRefineStage(BaseStage):
    name = "transcript_refine"

    def __init__(
        self,
        config: TranscriptPipelineConfig,
        audio_loader: Optional[AudioLoader] = None,
        sample_rate: int = 16000,
    ):
        self._cfg = config
        self._sample_rate = sample_rate
        self._audio_loader = audio_loader or AudioLoader(target_sample_rate=sample_rate)
        self._vtt = VTTParser()
        self._backends: List[BaseASRBackend] = []
        self._aligner = None
        self._ensembler = None

    def _ensure_loaded(self) -> None:
        if self._ensembler is not None:
            return
        for bcfg in self._cfg.enabled_backends():
            try:
                backend = build_asr_backend(bcfg.name, **bcfg.kwargs)
                backend.load(device=bcfg.device)
                self._backends.append(backend)
            except Exception as exc:
                logger.warning(f"transcript_refine: backend '{bcfg.name}' unavailable: {exc}")
        self._aligner = self._build_aligner()
        self._ensembler = ASREnsemblerFactory.build(
            strategy=self._cfg.ensemble_strategy,
            min_agree=self._cfg.min_agree,
            similarity_threshold=self._cfg.similarity_threshold,
            subtitle_weight=self._cfg.subtitle_weight,
        )

    def _build_aligner(self):
        name = (self._cfg.align_backend or "").strip().lower()
        if not name or name == "none":
            return None
        try:
            aligner = build_align_backend(name, **dict(self._cfg.align_kwargs))
            aligner.load(device=self._cfg.align_device)
            return aligner
        except Exception as exc:
            logger.warning(f"transcript_refine: forced aligner '{name}' unavailable: {exc}")
            return None

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        to_process, done = self._skip_processed(records, checkpoint)
        if not to_process:
            return done

        self._ensure_loaded()
        workers = getattr(self._cfg, "segment_concurrency", 1) or 1
        try:
            refined = parallel_map(self._refine_safe, to_process, workers)
        finally:
            self._free()
        for record in to_process:
            checkpoint.mark_processed(record["id"], self.name)
        checkpoint.save_state()
        return list(done) + refined

    def _refine_safe(self, record: Dict) -> Dict:
        try:
            return self._refine(record)
        except Exception as exc:
            logger.error(f"transcript_refine failed on {record['id']}: {exc}")
            return record

    def _refine(self, record: Dict) -> Dict:
        primary_text = record.get("text") or ""
        if not primary_text and not self._backends:
            return record

        audio, sr = self._audio_loader.load(
            record["audio_filepath"],
            offset=record.get("offset", 0.0),
            duration=record.get("duration"),
        )
        language = record.get("language")

        anchor_group = self._build_anchor_group(record, primary_text, language)
        hypotheses = anchor_group + self._run_backends(audio, sr, language)
        force_aligned = self._fill_word_timings(hypotheses, record)
        reference = self._subtitle_reference(record) if self._cfg.use_subtitle else None

        fused = self._ensembler.combine(hypotheses, reference=reference)

        record = dict(record)
        # Top-level so ConsensusStage's require_asr_agreement gate (reads
        # record["asr_confidence"]) can act on the ROVER word-agreement ratio.
        record["asr_confidence"] = float(fused.confidence)
        if fused.text:
            record["text"] = fused.text
            record["text_itn"] = fused.text
            if fused.word_timings:
                record["alignment"] = self._words_to_alignment(fused.word_timings)

        extra = dict(record.get("extra") or {})
        extra["asr_candidates"] = {h.backend: h.text for h in hypotheses if h.text}
        extra["asr_chosen"] = self._ensembler.name
        extra["asr_agreement"] = fused.confidence
        if force_aligned:
            extra["asr_force_aligned"] = force_aligned
        if isinstance(fused.raw, dict):
            extra["asr_words_changed"] = fused.raw.get("words_changed")
            extra["subtitle_used"] = fused.raw.get("subtitle_used", reference is not None)
        record["extra"] = extra
        return record

    def _build_anchor_group(
        self, record: Dict, primary_text: str, language: Optional[str]
    ) -> List[ASRResult]:
        extra = record.get("extra") or {}
        candidates = [
            c for c in (extra.get("asr_candidates_nbest") or [])
            if isinstance(c, str) and c.strip()
        ]
        if primary_text and (not candidates or candidates[0] != primary_text):
            candidates = [primary_text] + [c for c in candidates if c != primary_text]
        if not candidates:
            candidates = [primary_text]
        n = len(candidates)
        weight = (1.0 / n) if n > 1 else None
        group = [ASRResult(
            text=candidates[0],
            confidence=float(record.get("alignment_score") or 1.0),
            language=language,
            word_timings=self._alignment_to_words(record.get("alignment")),
            backend="primary",
            vote_weight=weight,
        )]
        for i, cand in enumerate(candidates[1:], start=1):
            group.append(ASRResult(
                text=cand,
                confidence=1.0,
                language=language,
                backend=f"primary_alt{i}",
                vote_weight=weight,
            ))
        return group

    def _run_backends(self, audio, sample_rate: int, language: Optional[str]) -> List[ASRResult]:
        def call(backend: BaseASRBackend) -> Optional[ASRResult]:
            try:
                return backend.transcribe(audio, sample_rate, language=language)
            except Exception as exc:
                logger.warning(f"ASR backend '{backend.name}' failed: {exc}")
                return None
        workers = getattr(self._cfg, "asr_concurrency", 1) or 1
        results = parallel_map(call, self._backends, workers)
        return [r for r in results if r is not None]

    def _fill_word_timings(self, hypotheses: List[ASRResult], record: Dict) -> List[str]:
        """Force-align text-only hypotheses (no word_timings) against the segment
        audio so they can join word-level ROVER. Returns the backends aligned."""
        if self._aligner is None:
            return []
        audio_path = record["audio_filepath"]
        lang = self._cfg.align_language
        targets = [h for h in hypotheses if not h.word_timings and h.text.strip()]

        def align_one(hyp: ASRResult) -> Optional[str]:
            try:
                result = self._aligner.align(audio_path, hyp.text, lang)
            except Exception as exc:
                logger.debug(f"forced align failed for {hyp.backend}: {exc}")
                return None
            if result.words:
                hyp.word_timings = [
                    WordTiming(word=w.text, start=float(w.start_time), end=float(w.end_time))
                    for w in result.words
                ]
                return hyp.backend
            return None

        workers = getattr(self._cfg, "asr_concurrency", 1) or 1
        filled = parallel_map(align_one, targets, workers)
        return [f for f in filled if f]

    def _subtitle_reference(self, record: Dict) -> Optional[ASRResult]:
        extra = record.get("extra") or {}
        path = extra.get("subtitle_path")
        start = record.get("start")
        end = record.get("end")
        if not path or start is None or end is None:
            return None
        try:
            words = self._vtt.words(path, float(start), float(end))
        except Exception as exc:
            logger.debug(f"subtitle parse failed for {record.get('id')}: {exc}")
            return None
        if not words:
            return None
        # Shift to segment-local time (anchor word timings are 0-based in the clip).
        off = float(start)
        timings = [
            WordTiming(word=w.word, start=max(0.0, w.start - off), end=max(0.0, w.end - off))
            for w in words
        ]
        return ASRResult(
            text=" ".join(w.word for w in words),
            confidence=1.0,
            word_timings=timings,
            backend="subtitle",
        )

    @staticmethod
    def _alignment_to_words(alignment) -> Optional[List[WordTiming]]:
        if not alignment:
            return None
        out: List[WordTiming] = []
        for a in alignment:
            token = (a.get("text") or "").strip()
            if not token:
                continue
            out.append(WordTiming(
                word=token,
                start=float(a.get("start_time", 0.0)),
                end=float(a.get("end_time", 0.0)),
            ))
        return out or None

    @staticmethod
    def _words_to_alignment(words: List[WordTiming]) -> List[Dict]:
        return [
            {"text": w.word, "start_time": float(w.start), "end_time": float(w.end)}
            for w in words
        ]

    def _free(self) -> None:
        for backend in self._backends:
            try:
                backend.unload()
            except Exception:
                pass
        if self._aligner is not None:
            try:
                self._aligner.unload()
            except Exception:
                pass
        self._backends = []
        self._aligner = None
        self._ensembler = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
