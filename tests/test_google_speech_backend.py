import datetime
import os

import numpy as np
import pytest

speech_v2 = pytest.importorskip("google.cloud.speech_v2")
from google.cloud.speech_v2.types import cloud_speech  # noqa: E402

from multitalker_asr.configs import GoogleSpeechConfig  # noqa: E402
from multitalker_asr.data.pipeline.asr_backends import (  # noqa: E402
    ASR_REGISTRY,
    GoogleSpeechBackend,
    build_asr_backend,
    list_asr_backends,
)


def _canned_response() -> cloud_speech.RecognizeResponse:
    return cloud_speech.RecognizeResponse(
        results=[
            cloud_speech.SpeechRecognitionResult(
                alternatives=[
                    cloud_speech.SpeechRecognitionAlternative(
                        transcript="xin chào",
                        confidence=0.9,
                        words=[
                            cloud_speech.WordInfo(
                                word="xin",
                                start_offset=datetime.timedelta(seconds=0.0),
                                end_offset=datetime.timedelta(seconds=0.4),
                                confidence=0.95,
                            ),
                            cloud_speech.WordInfo(
                                word="chào",
                                start_offset=datetime.timedelta(seconds=0.4),
                                end_offset=datetime.timedelta(seconds=0.8),
                                confidence=0.88,
                            ),
                        ],
                    )
                ]
            )
        ]
    )


class _FakeSpeechClient:
    last_request = None

    def __init__(self, *args, **kwargs):
        pass

    def recognize(self, request=None):
        _FakeSpeechClient.last_request = request
        return _canned_response()


@pytest.fixture
def fake_client(monkeypatch):
    monkeypatch.setattr(speech_v2, "SpeechClient", _FakeSpeechClient)
    _FakeSpeechClient.last_request = None
    return _FakeSpeechClient


class TestRegistry:
    def test_registered(self):
        assert "google_speech" in list_asr_backends()
        assert ASR_REGISTRY["google_speech"] is GoogleSpeechBackend

    def test_build(self):
        backend = build_asr_backend("google_speech", project_id="demo-project")
        assert isinstance(backend, GoogleSpeechBackend)
        assert backend.name == "google_speech"


class TestConfig:
    def test_resolves_project_from_env(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
        cfg = GoogleSpeechConfig()
        assert cfg.project_id == "env-project"
        assert cfg.model == "chirp_3"
        assert cfg.location == "us"
        assert cfg.recognizer_path == (
            "projects/env-project/locations/us/recognizers/_"
        )

    def test_chirp2_default_location(self):
        cfg = GoogleSpeechConfig(project_id="demo", model="chirp_2")
        assert cfg.location == "us-central1"

    def test_explicit_location_respected(self):
        cfg = GoogleSpeechConfig(project_id="demo", model="chirp_2", location="eu")
        assert cfg.location == "eu"

    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError, match="model must be one of"):
            GoogleSpeechConfig(project_id="demo", model="long")

    def test_missing_project_raises(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        with pytest.raises(ValueError, match="project_id"):
            GoogleSpeechConfig()

    def test_invalid_max_alternatives(self):
        with pytest.raises(ValueError, match="max_alternatives"):
            GoogleSpeechConfig(project_id="demo", max_alternatives=0)

    def test_empty_language_codes(self):
        with pytest.raises(ValueError, match="language_codes"):
            GoogleSpeechConfig(project_id="demo", language_codes=[])


class TestTranscribe:
    def test_transcribe_returns_result(self, fake_client):
        backend = build_asr_backend("google_speech", project_id="demo-project")
        backend.load()
        result = backend.transcribe(np.zeros(16000, np.float32), 16000, "vi-VN")
        assert result.text == "xin chào"
        assert result.backend == "google_speech"
        assert result.language == "vi-VN"
        assert result.confidence == pytest.approx(0.9)
        assert result.word_timings is not None
        assert len(result.word_timings) == 2
        assert result.word_timings[0].word == "xin"
        assert result.word_timings[1].end == pytest.approx(0.8)

    def test_single_alternative_has_no_raw_alternatives(self, fake_client):
        backend = build_asr_backend("google_speech", project_id="demo-project")
        backend.load()
        result = backend.transcribe(np.zeros(16000, np.float32), 16000, "vi-VN")
        assert "alternatives" not in result.raw

    def test_multiple_alternatives_in_raw(self, monkeypatch):
        def fake_recognize(self, request=None):
            return cloud_speech.RecognizeResponse(
                results=[
                    cloud_speech.SpeechRecognitionResult(
                        alternatives=[
                            cloud_speech.SpeechRecognitionAlternative(
                                transcript="xin chào", confidence=0.94
                            ),
                            cloud_speech.SpeechRecognitionAlternative(
                                transcript="xin chao", confidence=0.81
                            ),
                            cloud_speech.SpeechRecognitionAlternative(
                                transcript="sin chào", confidence=0.77
                            ),
                        ]
                    )
                ]
            )

        monkeypatch.setattr(speech_v2, "SpeechClient", _FakeSpeechClient)
        monkeypatch.setattr(_FakeSpeechClient, "recognize", fake_recognize)
        backend = build_asr_backend(
            "google_speech", project_id="demo-project", max_alternatives=3
        )
        backend.load()
        result = backend.transcribe(np.zeros(16000, np.float32), 16000, "vi-VN")
        assert result.text == "xin chào"
        alts = result.raw["alternatives"]
        assert len(alts) == 3
        assert alts[0]["transcript"] == "xin chào"
        assert alts[2]["confidence"] == pytest.approx(0.77)

    def test_transcribe_requires_load(self):
        backend = build_asr_backend("google_speech", project_id="demo-project")
        with pytest.raises(Exception, match="not loaded"):
            backend.transcribe(np.zeros(16000, np.float32), 16000)

    def test_request_uses_recognizer_path(self, fake_client):
        backend = build_asr_backend("google_speech", project_id="demo-project")
        backend.load()
        backend.transcribe(np.zeros(16000, np.float32), 16000, "vi-VN")
        request = _FakeSpeechClient.last_request
        assert request.recognizer == (
            "projects/demo-project/locations/us/recognizers/_"
        )
        assert list(request.config.language_codes) == ["vi-VN"]


class TestConfigBuilding:
    def test_features_mapped(self, fake_client):
        backend = build_asr_backend(
            "google_speech",
            project_id="demo-project",
            model="chirp_2",
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
        )
        backend.load()
        config = backend._build_recognition_config("vi-VN")
        assert config.model == "chirp_2"
        assert config.features.enable_automatic_punctuation is True
        assert config.features.enable_word_time_offsets is True

    def test_chirp3_disables_word_features(self, fake_client):
        backend = build_asr_backend(
            "google_speech",
            project_id="demo-project",
            model="chirp_3",
            enable_word_time_offsets=True,
            enable_word_confidence=True,
        )
        backend.load()
        config = backend._build_recognition_config("vi-VN")
        assert config.features.enable_word_time_offsets is False
        assert config.features.enable_word_confidence is False

    def test_phrase_set_adaptation_mapped(self, fake_client):
        backend = build_asr_backend(
            "google_speech",
            project_id="demo-project",
            phrase_sets=[{"value": "NeMo", "boost": 15.0}],
        )
        backend.load()
        config = backend._build_recognition_config(None)
        assert len(config.adaptation.phrase_sets) == 1
        assert config.adaptation.phrase_sets[0].inline_phrase_set.phrases[0].value == (
            "NeMo"
        )


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_CLOUD_PROJECT"),
    reason="requires GOOGLE_CLOUD_PROJECT and live Google credentials",
)
class TestLive:
    def test_live_transcribe(self):
        sample = os.environ.get("GOOGLE_SPEECH_SAMPLE_WAV")
        if not sample or not os.path.exists(sample):
            pytest.skip("set GOOGLE_SPEECH_SAMPLE_WAV to a local wav file")
        import soundfile as sf

        audio, sr = sf.read(sample, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        backend = build_asr_backend("google_speech", language_codes=["en-US"])
        backend.load()
        result = backend.transcribe(audio, sr)
        assert result.backend == "google_speech"
        assert isinstance(result.text, str)
