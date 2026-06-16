import numpy as np
import pytest

from multitalker_asr.data.pipeline.vad_backends import (
    VAD_REGISTRY,
    BaseVADBackend,
    FsmnVADBackend,
    VADResult,
    VADSegment,
    assemble_frames,
    build_vad_backend,
    list_vad_backends,
    merge_segments,
    register_vad_backend,
)


class TestVADResult:
    def test_from_spans_computes_ratio(self):
        result = VADResult.from_spans([(0.0, 1.0), (2.0, 2.5)], total_duration=5.0)
        assert len(result.segments) == 2
        assert result.speech_ratio == pytest.approx(1.5 / 5.0)

    def test_from_spans_empty(self):
        result = VADResult.from_spans([], total_duration=5.0)
        assert result.segments == []
        assert result.speech_ratio == 0.0

    def test_segment_duration(self):
        assert VADSegment(1.0, 2.5).duration == pytest.approx(1.5)


class TestMergeSegments:
    def test_merges_within_gap(self):
        merged = merge_segments(
            [VADSegment(0.0, 1.0), VADSegment(1.1, 2.0)], min_gap_s=0.2
        )
        assert len(merged) == 1
        assert merged[0].start == pytest.approx(0.0)
        assert merged[0].end == pytest.approx(2.0)

    def test_keeps_separated(self):
        merged = merge_segments(
            [VADSegment(0.0, 1.0), VADSegment(3.0, 4.0)], min_gap_s=0.2
        )
        assert len(merged) == 2

    def test_drops_below_min_dur(self):
        merged = merge_segments(
            [VADSegment(0.0, 0.05), VADSegment(3.0, 4.0)], min_gap_s=0.2, min_dur_s=0.1
        )
        assert len(merged) == 1
        assert merged[0].start == pytest.approx(3.0)

    def test_padding_clamps_at_zero(self):
        merged = merge_segments([VADSegment(0.05, 1.0)], pad_s=0.2)
        assert merged[0].start == pytest.approx(0.0)


class TestAssembleFrames:
    def test_flags_to_segments(self):
        flags = [False, True, True, False, False, True, False]
        segs = assemble_frames(flags, hop_size=160, sample_rate=16000, min_gap_s=0.0)
        frame = 160 / 16000
        assert len(segs) == 2
        assert segs[0].start == pytest.approx(1 * frame)
        assert segs[0].end == pytest.approx(3 * frame)

    def test_trailing_speech_closed(self):
        flags = [False, True, True]
        segs = assemble_frames(flags, hop_size=160, sample_rate=16000, min_gap_s=0.0)
        assert len(segs) == 1
        assert segs[0].end == pytest.approx(3 * 160 / 16000)


class TestRegistry:
    def test_registered(self):
        names = list_vad_backends()
        assert {"silero", "fsmn", "ten", "pyannote_seg"} <= set(names)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown VAD backend"):
            build_vad_backend("nope")

    def test_custom_registration(self):
        class _Stub(BaseVADBackend):
            name = "_stub_vad"

            def load(self, device="cpu"):
                self._loaded = True

            def detect(self, audio, sample_rate):
                return VADResult.from_spans([(0.0, 1.0)], len(audio) / sample_rate)

        register_vad_backend("_stub_vad", _Stub)
        try:
            backend = build_vad_backend("_stub_vad")
            backend.load()
            out = backend.detect(np.zeros(16000, np.float32), 16000)
            assert out.segments[0].end == pytest.approx(1.0)
        finally:
            VAD_REGISTRY.pop("_stub_vad", None)


class TestFsmnMsConversion:
    def test_ms_to_seconds(self):
        class _FakeModel:
            def generate(self, input):
                return [{"value": [[0, 1000], [2000, 2500]]}]

        backend = FsmnVADBackend()
        backend._model = _FakeModel()
        backend._loaded = True
        out = backend.detect(np.zeros(48000, np.float32), 16000)
        assert out.segments[0].start == pytest.approx(0.0)
        assert out.segments[0].end == pytest.approx(1.0)
        assert out.segments[1].end == pytest.approx(2.5)
        assert out.speech_ratio == pytest.approx(1.5 / 3.0)
