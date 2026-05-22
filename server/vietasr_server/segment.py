"""VAD pre-segmentation for long batch audio.

A long clip is transcribed better by first finding speech regions with a neural
VAD, cutting the audio into <=60 s pieces *at silence boundaries*, and running
ASR on each piece. This skips silence, never cuts mid-word, avoids the encoder's
mid-stream re-anchoring (positional-encoding window ~3.6 min), and keeps every
ASR call well within the model's length limit.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from vietasr_server.asr import AsrEngine, EngineRegistry

# Auto-segment batch audio longer than this (seconds).
AUTO_SEGMENT_THRESHOLD_S = 180.0
# Maximum length of one ASR chunk (seconds).
MAX_SEGMENT_S = 60.0
# Preset used to find speech regions — a real neural VAD (Silero).
VAD_PRESET = "vad"


def _has_asr(engine: AsrEngine) -> bool:
    """True if the engine's pipeline transcribes (segmentation only helps ASR)."""
    if engine.spec[0] == "modules":
        return "vietasr" in engine.spec[1:]
    return engine.spec[1] != "vad"  # every preset except 'vad' transcribes


def should_segment(mode: str, duration_s: float, engine: AsrEngine) -> bool:
    """Decide whether to run the VAD pre-segmentation path."""
    mode = (mode or "auto").lower()
    if mode == "off":
        return False
    if mode == "on":
        return True
    # auto: only for long clips on a transcribing pipeline.
    return _has_asr(engine) and duration_s > AUTO_SEGMENT_THRESHOLD_S


def group_regions(regions: List[dict],
                  max_segment_s: float = MAX_SEGMENT_S) -> List[Tuple[float, float]]:
    """Pack voiced regions into chunks of <= max_segment_s, cutting only at
    silence (region boundaries). A single region longer than the limit is
    hard-split. Returns a list of (start_s, end_s) chunk spans."""
    chunks: List[Tuple[float, float]] = []
    cur_start = None
    cur_end = 0.0
    for r in regions:
        rs, re = float(r["start"]), float(r["end"])
        if re <= rs:
            continue
        if re - rs > max_segment_s:
            # A single region longer than the limit — flush, then hard-split it.
            if cur_start is not None:
                chunks.append((cur_start, cur_end))
                cur_start = None
            t = rs
            while t < re:
                chunks.append((t, min(t + max_segment_s, re)))
                t += max_segment_s
            continue
        if cur_start is None:
            cur_start, cur_end = rs, re
        elif re - cur_start <= max_segment_s:
            cur_end = re
        else:
            chunks.append((cur_start, cur_end))
            cur_start, cur_end = rs, re
    if cur_start is not None:
        chunks.append((cur_start, cur_end))
    return chunks


def transcribe_segmented(registry: EngineRegistry, asr_engine: AsrEngine,
                         pcm: np.ndarray, sample_rate: float,
                         max_segment_s: float = MAX_SEGMENT_S) -> dict:
    """VAD -> cut into <=max_segment_s chunks at silence -> ASR per chunk ->
    merge. Returns a result payload (text, segments, voiced_regions, speech,
    chunks)."""
    # 1. Neural VAD over the whole clip.
    vad = registry.resolve(preset=VAD_PRESET).transcribe(pcm, sample_rate)
    regions = vad.payload.get("voiced_regions", [])
    speech = vad.payload.get("speech", {})

    # 2. Group voiced regions into <=max_segment_s chunks.
    chunks = group_regions(regions, max_segment_s)
    if not chunks:
        return {"text": "", "segments": [], "voiced_regions": regions,
                "speech": speech, "chunks": []}

    # 3. ASR per chunk; 4. merge (offset chunk-local segment times).
    texts: List[str] = []
    segments: List[dict] = []
    spans: List[dict] = []
    n = len(pcm)
    for start_s, end_s in chunks:
        a = max(0, int(start_s * sample_rate))
        b = min(n, int(end_s * sample_rate))
        if b <= a:
            continue
        result = asr_engine.transcribe(pcm[a:b], sample_rate)
        if result.text:
            texts.append(result.text)
        for seg in result.segments:
            seg = dict(seg)
            seg["start"] = float(seg.get("start", 0.0)) + start_s
            seg["end"] = float(seg.get("end", 0.0)) + start_s
            segments.append(seg)
        spans.append({"start": round(start_s, 3), "end": round(end_s, 3)})

    return {
        "text": " ".join(texts),
        "segments": segments,
        "voiced_regions": regions,
        "speech": speech,
        "chunks": spans,
    }
