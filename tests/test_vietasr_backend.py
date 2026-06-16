import numpy as np
import pytest

from multitalker_asr.data.pipeline.asr_backends import build_asr_backend, list_asr_backends
from multitalker_asr.data.pipeline.asr_backends.vietasr import VietASRBackend


class _FakeResult:
    def __init__(self, text, segments=None):
        self.text = text
        self.segments = segments or []


class _FakePipeline:
    def __init__(self):
        self.received = None

    def transcribe(self, source, sample_rate=16000.0):
        self.received = (source, sample_rate)
        return _FakeResult("xin chào các bạn")

    def close(self):
        pass


class TestVietASRRegistry:
    def test_registered(self):
        assert "vietasr" in list_asr_backends()
        assert isinstance(build_asr_backend("vietasr"), VietASRBackend)


class TestVietASRHelpers:
    def test_to_int16_from_float(self):
        b = VietASRBackend()
        out = b._to_int16(np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32))
        assert out.dtype == np.int16
        assert out[1] == 32767 and out[2] == -32767

    def test_to_int16_passthrough(self):
        b = VietASRBackend()
        out = b._to_int16(np.array([1, 2, 3], dtype=np.int16))
        assert out.dtype == np.int16 and list(out) == [1, 2, 3]

    def test_parse_segments(self):
        b = VietASRBackend()
        wt = b._parse_segments([{"text": "xin", "start": 0.0, "end": 0.3}])
        assert wt[0].word == "xin" and wt[0].end == 0.3
        assert b._parse_segments([]) is None


class TestVietASRTranscribe:
    def test_transcribe_uses_int16_and_text(self):
        b = VietASRBackend()
        fake = _FakePipeline()
        b._pipeline = fake
        b._loaded = True
        res = b.transcribe(np.zeros(16000, dtype=np.float32), 16000)
        assert res.text == "xin chào các bạn"
        assert res.backend == "vietasr"
        assert res.language == "vi"
        assert fake.received[0].dtype == np.int16
        assert fake.received[1] == 16000.0
