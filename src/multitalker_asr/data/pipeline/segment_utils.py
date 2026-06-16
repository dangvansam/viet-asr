from typing import Dict, List, Optional, Tuple


def interval_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Intersection-over-union of two [start, end] intervals in seconds."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def spans_to_pairs(segments: List[Dict]) -> List[Tuple[float, float]]:
    return [(float(s["start"]), float(s["end"])) for s in segments]


def speech_span(
    segments: List[Tuple[float, float]],
    pad_s: float = 0.0,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """Outer bound of speech regions, padded and clamped to [lo, hi] when given."""
    if not segments:
        return None
    start = min(s for s, _ in segments) - pad_s
    end = max(e for _, e in segments) + pad_s
    if lo is not None:
        start = max(start, lo)
    if hi is not None:
        end = min(end, hi)
    if end <= start:
        return None
    return start, end
