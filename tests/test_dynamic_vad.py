import numpy as np
import pytest

from multitalker_asr.data.pipeline.vad_backends import (
    VADBackendError,
    VADSegment,
    build_vad_backend,
    dynamic_merge_segments,
    list_vad_backends,
    lookup_silence_s,
    normalize_silence_schedule,
)


def test_normalize_large_tail_is_effectively_infinite():
    sched = normalize_silence_schedule([[5000, 1500], [1e9, 300]])
    assert sched[0] == (5000.0, 1500.0)
    assert lookup_silence_s(1e6, sched) == pytest.approx(0.3)


def test_normalize_inf_tail_from_none():
    sched = normalize_silence_schedule([[5000, 1500], [None, 300]])
    assert sched[-1][0] == float("inf")
    assert sched[-1][1] == 300.0


def test_normalize_sorts_ascending():
    sched = normalize_silence_schedule([[1e9, 100], [5000, 1500], [20000, 800]])
    limits = [limit for limit, _ in sched]
    assert limits == sorted(limits)


def test_normalize_empty_raises():
    with pytest.raises(VADBackendError):
        normalize_silence_schedule([])


def test_lookup_shrinks_with_accumulation():
    sched = normalize_silence_schedule([[5000, 1500], [1e9, 300]])
    assert lookup_silence_s(2.0, sched) == pytest.approx(1.5)
    assert lookup_silence_s(60.0, sched) == pytest.approx(0.3)


def test_short_gap_merges_when_low_accumulated():
    segs = [VADSegment(0.0, 1.0), VADSegment(1.4, 2.4)]
    out = dynamic_merge_segments(segs, [[5000, 1500], [1e9, 300]])
    assert len(out) == 1
    assert out[0].start == pytest.approx(0.0)
    assert out[0].end == pytest.approx(2.4)


def test_same_gap_cuts_when_high_accumulated():
    segs = [VADSegment(0.0, 6.0), VADSegment(6.4, 7.4)]
    out = dynamic_merge_segments(segs, [[5000, 1500], [1e9, 300]])
    assert len(out) == 2
    assert out[0].end == pytest.approx(6.0)
    assert out[1].start == pytest.approx(6.4)


def test_empty_returns_empty():
    assert dynamic_merge_segments([], [[5000, 1500], [1e9, 300]]) == []


def test_min_dur_filter_drops_short_segment():
    segs = [VADSegment(0.0, 0.05)]
    out = dynamic_merge_segments(segs, [[1e9, 300]], min_dur_s=0.1)
    assert out == []


def test_overlapping_segments_never_produce_overlap():
    segs = [VADSegment(0.0, 2.0), VADSegment(1.5, 3.0)]
    out = dynamic_merge_segments(segs, [[1e9, 100]])
    assert len(out) == 1
    assert out[0].end == pytest.approx(3.0)


def test_dynamic_backend_registered():
    assert "dynamic" in list_vad_backends()


def test_dynamic_backend_constructs():
    backend = build_vad_backend(
        "dynamic", provider="silero", silence_schedule=[[5000, 1500], [1e9, 300]]
    )
    assert backend.name == "dynamic"


def test_dynamic_backend_empty_schedule_falls_back_to_default():
    backend = build_vad_backend("dynamic", provider="silero", silence_schedule=[])
    assert backend.name == "dynamic"
    assert backend._schedule[-1][1] == pytest.approx(100.0)


def _silero_available() -> bool:
    try:
        import silero_vad  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _silero_available(), reason="silero_vad not installed")
def test_dynamic_wrapper_over_silero_cpu():
    backend = build_vad_backend(
        "dynamic", provider="silero", silence_schedule=[[5000, 1500], [1e9, 300]]
    )
    backend.load(device="cpu")
    audio = np.zeros(16000 * 3, dtype=np.float32)
    result = backend.detect(audio, 16000)
    backend.unload()
    assert result.backend == "dynamic"
    assert result.raw["provider"] == "silero"
    assert "base_segments" in result.raw and "dynamic_segments" in result.raw
