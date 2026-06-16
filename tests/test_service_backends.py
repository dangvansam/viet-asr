"""URL-only service-client backends: assert HTTP contract + parsing (no server)."""

import numpy as np

from multitalker_asr.data.pipeline.align_backends import build_align_backend
from multitalker_asr.data.pipeline.vad_backends import build_vad_backend


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self._payload = payload
        self.last_call = {}

    def post(self, url, files=None, data=None, timeout=None):
        self.last_call = {"url": url, "files": files, "data": data}
        return _Resp(self._payload)

    def close(self):
        return None


def test_service_vad_posts_to_vad_endpoint_and_parses(monkeypatch):
    payload = {
        "segments": [{"start": 0.5, "end": 1.5}, {"start": 2.0, "end": 3.0}],
        "speech_ratio": 0.62,
        "backend": "silero",
        "raw": {"n_providers": 1},
    }
    backend = build_vad_backend("service", base_url="http://vad:9001")
    backend.load()
    session = _FakeSession(payload)
    backend._session = session

    result = backend.detect(np.zeros(16000, np.float32), 16000)

    assert session.last_call["url"] == "http://vad:9001/v1/audio/vad"
    assert "file" in session.last_call["files"]
    assert [(s.start, s.end) for s in result.segments] == [(0.5, 1.5), (2.0, 3.0)]
    assert result.speech_ratio == 0.62
    assert result.backend == "silero"
    assert result.raw == {"n_providers": 1}


def test_service_vad_empty_segments(monkeypatch):
    backend = build_vad_backend("service", base_url="http://vad:9001")
    backend.load()
    backend._session = _FakeSession({"segments": [], "speech_ratio": 0.0})
    result = backend.detect(np.zeros(8000, np.float32), 16000)
    assert result.segments == []
    assert result.speech_ratio == 0.0


def test_service_align_posts_to_align_endpoint_and_parses(tmp_path):
    wav = tmp_path / "a.wav"
    import soundfile as sf

    sf.write(str(wav), np.zeros(16000, np.float32), 16000)
    payload = {"task": "align", "words": [
        {"word": "xin", "start": 0.0, "end": 0.5},
        {"word": "chào", "start": 0.5, "end": 1.0},
    ]}
    backend = build_align_backend("service", base_url="http://align-mms:9203")
    backend.load()
    backend._session = _FakeSession(payload)

    result = backend.align(str(wav), "xin chào", "Vietnamese")

    assert backend._session.last_call["url"] == "http://align-mms:9203/v1/audio/alignments"
    assert [w.text for w in result.words] == ["xin", "chào"]
    assert result.span == (0.0, 1.0)
