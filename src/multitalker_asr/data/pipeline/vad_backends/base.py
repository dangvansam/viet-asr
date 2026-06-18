import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class VADSegment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class VADResult:
    segments: List[VADSegment]
    speech_ratio: float = 0.0
    backend: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_spans(
        cls,
        spans: List[tuple],
        total_duration: float,
        backend: str = "",
        raw: Dict[str, Any] = None,
    ) -> "VADResult":
        segments = [VADSegment(float(s), float(e)) for s, e in spans]
        speech = sum(seg.duration for seg in segments)
        ratio = speech / total_duration if total_duration > 0 else 0.0
        return cls(segments=segments, speech_ratio=ratio, backend=backend, raw=raw or {})


class BaseVADBackend(ABC):
    name: str = ""

    @abstractmethod
    def load(self, device: str = "cpu") -> None:
        ...

    @abstractmethod
    def detect(self, audio: np.ndarray, sample_rate: int) -> VADResult:
        ...

    def unload(self) -> None:
        return None

    @property
    def is_loaded(self) -> bool:
        return getattr(self, "_loaded", False)


class VADBackendError(RuntimeError):
    pass


def merge_segments(
    segments: List[VADSegment],
    min_gap_s: float = 0.2,
    min_dur_s: float = 0.0,
    pad_s: float = 0.0,
) -> List[VADSegment]:
    """Sort, pad, merge segments closer than min_gap_s, drop those below min_dur_s."""
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: s.start)
    merged: List[VADSegment] = []
    for seg in ordered:
        start = max(0.0, seg.start - pad_s)
        end = seg.end + pad_s
        if merged and start - merged[-1].end <= min_gap_s:
            merged[-1].end = max(merged[-1].end, end)
        else:
            merged.append(VADSegment(start, end))

    return [seg for seg in merged if seg.duration >= min_dur_s]


def assemble_frames(
    flags: List[bool],
    hop_size: int,
    sample_rate: int,
    min_gap_s: float = 0.2,
    min_dur_s: float = 0.0,
) -> List[VADSegment]:
    """Convert per-frame speech flags into merged second-domain segments."""
    frame_dur = hop_size / sample_rate
    spans: List[VADSegment] = []
    in_speech = False
    start_idx = 0
    for idx, flag in enumerate(flags):
        if flag and not in_speech:
            in_speech = True
            start_idx = idx
        elif not flag and in_speech:
            in_speech = False
            spans.append(VADSegment(start_idx * frame_dur, idx * frame_dur))
    if in_speech:
        spans.append(VADSegment(start_idx * frame_dur, len(flags) * frame_dur))

    return merge_segments(spans, min_gap_s=min_gap_s, min_dur_s=min_dur_s)


def normalize_silence_schedule(
    schedule: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Coerce a YAML/JSON silence schedule into a sorted list of float tuples.

    Each entry is ``(accumulated_ms_limit, silence_threshold_ms)``. A ``None`` or
    non-finite final limit (YAML has no ``inf``) is treated as ``float('inf')`` so the
    last bucket catches all longer accumulations. Sorted ascending by limit.
    """
    if not schedule:
        raise VADBackendError("silence_schedule is empty")
    normalized: List[Tuple[float, float]] = []
    for limit, silence in schedule:
        if limit is None or not math.isfinite(float(limit)):
            limit_ms = float("inf")
        else:
            limit_ms = float(limit)
        normalized.append((limit_ms, float(silence)))
    return sorted(normalized, key=lambda x: x[0])


def lookup_silence_s(accumulated_s: float, schedule: List[Tuple[float, float]]) -> float:
    """Silence threshold (seconds) for the current accumulated speech (seconds).

    The threshold shrinks as accumulated speech grows: the first bucket whose
    ``limit_ms`` is >= the accumulated duration wins. ``schedule`` must already be
    normalized (sorted ascending by limit).
    """
    accumulated_ms = accumulated_s * 1000.0
    for limit_ms, silence_ms in schedule:
        if accumulated_ms <= limit_ms:
            return silence_ms / 1000.0
    return schedule[-1][1] / 1000.0


def dynamic_merge_segments(
    segments: List[VADSegment],
    silence_schedule: List[Tuple[float, float]],
    min_dur_s: float = 0.0,
    pad_s: float = 0.0,
) -> List[VADSegment]:
    """Re-cut/merge segments with a duration-adaptive silence threshold.

    Walks segments left-to-right accumulating speech since the last cut. A gap is
    merged when it is shorter than the schedule's threshold for the current
    accumulation (short utterances tolerate longer silences), otherwise it cuts
    (long runs cut on even short silences). Pure function — no models, no I/O.
    """
    if not segments:
        return []

    sched = normalize_silence_schedule(silence_schedule)
    ordered = sorted(segments, key=lambda s: s.start)

    merged: List[VADSegment] = []
    current = VADSegment(ordered[0].start, ordered[0].end)
    accumulated = current.duration
    for seg in ordered[1:]:
        gap = seg.start - current.end
        if gap <= lookup_silence_s(accumulated, sched):
            current.end = max(current.end, seg.end)
            accumulated += seg.duration
        else:
            merged.append(current)
            current = VADSegment(seg.start, seg.end)
            accumulated = seg.duration
    merged.append(current)

    return merge_segments(merged, min_gap_s=0.0, min_dur_s=min_dur_s, pad_s=pad_s)
