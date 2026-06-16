import math

import numpy as np
import pytest

from multitalker_asr.data.pipeline.asr_backends import (
    ASR_REGISTRY,
    OpenAITranscriptionBackend,
    build_asr_backend,
    list_asr_backends,
)


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self._payload = payload
        self.last_call = None

    def post(self, url, **kwargs):
        self.last_call = {"url": url, **kwargs}
        return _FakeResponse(self._payload)

    def close(self):
        return None


def _backend(payload, **kwargs):
    backend = build_asr_backend("openai_transcription", base_url="http://h:8101", **kwargs)
    backend.load()
    backend._session = _FakeSession(payload)
    return backend


class TestRegistry:
    def test_registered_with_alias(self):
        names = list_asr_backends()
        assert "openai_transcription" in names
        assert "openai" in names
        assert ASR_REGISTRY["openai"] is OpenAITranscriptionBackend

    def test_transcribe_requires_load(self):
        backend = build_asr_backend("openai_transcription")
        with pytest.raises(Exception, match="not loaded"):
            backend.transcribe(np.zeros(16000, np.float32), 16000)


class TestRequest:
    def test_targets_transcriptions_endpoint(self):
        backend = _backend({"text": "ok"}, model="m", diarization=True)
        backend.transcribe(np.zeros(16000, np.float32), 16000, "vi")
        call = backend._session.last_call
        assert call["url"] == "http://h:8101/v1/audio/transcriptions"
        assert "file" in call["files"]
        data = call["data"]
        assert ("response_format", "verbose_json") in data
        assert ("language", "vi") in data
        assert ("model", "m") in data
        assert ("diarization", "true") in data
        assert ("timestamp_granularities[]", "word") in data
        assert ("include[]", "logprobs") in data


class TestNBestCandidates:
    def test_dict_candidates_normalized_to_strings(self):
        payload = {"text": "a b c", "candidates": [{"text": "a b c"}, {"text": "a b d"}]}
        backend = _backend(payload, n_best=2)
        res = backend.transcribe(np.zeros(16000, np.float32), 16000, "vi")
        assert res.raw["candidates"] == ["a b c", "a b d"]
        assert all(isinstance(c, str) for c in res.raw["candidates"])

    def test_single_candidate_drops_raw_candidates(self):
        payload = {"text": "a b c", "candidates": [{"text": "a b c"}]}
        backend = _backend(payload, n_best=2)
        res = backend.transcribe(np.zeros(16000, np.float32), 16000, "vi")
        assert "candidates" not in res.raw

    def test_nbest_request_params_sent(self):
        backend = _backend({"text": "ok"}, n_best=3, nbest_temperature=0.3)
        backend.transcribe(np.zeros(16000, np.float32), 16000, "vi")
        data = backend._session.last_call["data"]
        assert ("n_best", "3") in data
        assert ("nbest_temperature", "0.3") in data


class TestParse:
    def test_top_level_words(self):
        payload = {
            "text": "xin chào",
            "language": "vi",
            "words": [
                {"word": "xin", "start": 0.0, "end": 0.4},
                {"word": "chào", "start": 0.4, "end": 0.8},
            ],
        }
        r = _backend(payload).transcribe(np.zeros(16000, np.float32), 16000)
        assert r.text == "xin chào"
        assert r.language == "vi"
        assert r.backend == "openai_transcription"
        assert len(r.word_timings) == 2
        assert r.word_timings[1].end == pytest.approx(0.8)

    def test_segment_only_words_flattened(self):
        payload = {
            "text": "a b",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "a b",
                    "speaker": "spk0",
                    "words": [
                        {"word": "a", "start": 0.0, "end": 0.5},
                        {"word": "b", "start": 0.5, "end": 1.0},
                    ],
                }
            ],
        }
        r = _backend(payload).transcribe(np.zeros(16000, np.float32), 16000)
        assert len(r.word_timings) == 2
        assert r.raw["segments"][0]["speaker"] == "spk0"

    def test_logprob_to_confidence(self):
        payload = {
            "text": "hi",
            "words": [
                {"word": "hi", "start": 0.0, "end": 0.4, "logprob": math.log(0.5)},
            ],
        }
        r = _backend(payload).transcribe(np.zeros(16000, np.float32), 16000)
        assert r.confidence == pytest.approx(0.5)
        assert r.word_timings[0].confidence == pytest.approx(0.5)

    def test_no_logprob_defaults_confidence_one(self):
        r = _backend({"text": "hi", "words": []}).transcribe(
            np.zeros(16000, np.float32), 16000
        )
        assert r.confidence == 1.0


class _FallbackSession:
    """400 for verbose_json, 200 plain json — mimics vLLM Qwen3-ASR."""

    def __init__(self):
        self.formats = []

    def post(self, url, **kwargs):
        fmt = dict(kwargs["data"]).get("response_format")
        self.formats.append(fmt)
        if fmt == "verbose_json":
            return _FakeResponse({"error": "unsupported"}, status_code=400)
        return _FakeResponse({"text": "mời các bạn"}, status_code=200)

    def close(self):
        return None


def test_verbose_json_falls_back_to_json():
    backend = build_asr_backend("openai_transcription", base_url="http://h:8101")
    backend.load()
    session = _FallbackSession()
    backend._session = session
    r = backend.transcribe(np.zeros(16000, np.float32), 16000, "vi")
    assert session.formats == ["verbose_json", "json"]
    assert r.text == "mời các bạn"
    assert r.word_timings is None
