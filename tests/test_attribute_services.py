"""Speaker (/v1/audio/embeddings) + Gender (/v1/audio/classifications) CLIENTS.

The server side now runs as standalone LitServe services in the external
speaker-recognition / gender-classification repos; here we only test this repo's
clients (request path, legacy fallback, response parsing).
"""

import numpy as np
import pytest

from multitalker_asr.data.pipeline.speaker.embedder import SpeakerEmbedder
from multitalker_asr.data.pipeline.stages.gender_classify import GenderClassifyStage


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self._payload = payload
        self.last = None

    def post(self, url, **kw):
        self.last = {"url": url, **kw}
        return _FakeResp(self._payload)

    def close(self):
        return None


class TestSpeakerEmbedderClient:
    def test_posts_v1_endpoint_and_parses(self, tmp_path):
        wav = tmp_path / "a.wav"
        import soundfile as sf

        sf.write(str(wav), np.zeros(16000, np.float32), 16000)
        emb = SpeakerEmbedder(url="http://spk:2010")
        emb._session = _FakeSession({"object": "audio.embedding", "embedding": [1.0, 2.0, 3.0]})
        vec = emb.embed(str(wav))
        assert emb._session.last["url"] == "http://spk:2010/v1/audio/embeddings"
        assert vec.tolist() == [1.0, 2.0, 3.0]

    def test_legacy_url_and_nested_embedding(self, tmp_path):
        wav = tmp_path / "a.wav"
        import soundfile as sf

        sf.write(str(wav), np.zeros(16000, np.float32), 16000)
        emb = SpeakerEmbedder(url="http://spk:2010/embed")  # legacy endpoint URL
        emb._session = _FakeSession({"success": True, "data": {"embedding": [4.0, 5.0]}})
        vec = emb.embed(str(wav))
        assert emb._session.last["url"] == "http://spk:2010/v1/audio/embeddings"
        assert vec.tolist() == [4.0, 5.0]


class TestGenderParse:
    def test_openai_shape(self):
        out = GenderClassifyStage._parse(
            {"label": "MALE", "confidence": 0.8, "probs": {"male": 0.8, "female": 0.2}}
        )
        assert out == {"gender": "male", "gender_confidence": 0.8}

    def test_legacy_shape(self):
        out = GenderClassifyStage._parse({"gender": "FEMALE", "probs": [0.3, 0.7]})
        assert out["gender"] == "female"
        assert out["gender_confidence"] == pytest.approx(0.7)

    def test_base_url_strip(self):
        assert GenderClassifyStage._base_url("http://g:8000/predict") == "http://g:8000"
        assert GenderClassifyStage._base_url("http://g:8000/v1/audio/classifications") == "http://g:8000"
        assert GenderClassifyStage._base_url("http://g:8000") == "http://g:8000"
