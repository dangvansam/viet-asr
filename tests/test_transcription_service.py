"""TranscriptionLitAPI — decode/predict/envelope contract (no server needed)."""

import io
import time

import numpy as np
import pytest

pytest.importorskip("litserve")

from multitalker_asr.data.pipeline.asr_backends.base import ASRResult, WordTiming  # noqa: E402
from multitalker_asr.serving import TranscriptionLitAPI  # noqa: E402


def _result(language="vi"):
    return ASRResult(
        text="xin chào",
        confidence=0.5,
        language=language,
        word_timings=[
            WordTiming(word="xin", start=0.0, end=0.4),
            WordTiming(word="chào", start=0.4, end=0.8),
        ],
        backend="stub",
    )


class _StubLoader:
    def load(self, path):
        return np.zeros(16000, np.float32), 16000


class _BatchBackend:
    def transcribe_batch(self, audios, sr, language):
        return [_result(language or "vi") for _ in audios]


class _SingleBackend:
    def transcribe(self, audio, sr, language):
        return _result(language or "vi")


def _api(backend, has_batch, language=None):
    api = TranscriptionLitAPI("stub", {}, language=language, model_id="stub-1")
    api._backend = backend
    api._loader = _StubLoader()
    api._has_batch = has_batch
    return api


def _item(language=None, granularities=None, response_format="verbose_json"):
    params = {"response_format": response_format}
    if language:
        params["language"] = language
    if granularities:
        params["timestamp_granularities"] = granularities
    return {"path": "/nonexistent.wav", "params": params, "t0": time.perf_counter()}


def test_predict_batch_groups_by_language():
    api = _api(_BatchBackend(), has_batch=True)
    out = api.predict([_item("vi"), _item("en")])
    assert [r["output"].language for r in out] == ["vi", "en"]
    assert out[0]["params"]["language"] == "vi"
    assert "elapsed" in out[0]


def test_predict_single_backend_loops():
    api = _api(_SingleBackend(), has_batch=False, language="vi")
    out = api.predict([_item(), _item()])
    assert all(r["output"].text == "xin chào" for r in out)


def test_envelope_verbose_json_with_words():
    api = _api(_BatchBackend(), has_batch=True)
    body = api.encode_response(
        {"output": _result(), "params": {"timestamp_granularities": ["word"]}, "elapsed": 0.01}
    )
    assert body["task"] == "transcribe"
    assert body["text"] == "xin chào"
    assert body["language"] == "vi"
    assert len(body["words"]) == 2
    assert body["segments"][0]["words"][1]["end"] == pytest.approx(0.8)
    assert body["elapsed_s"] == 0.01


def test_envelope_segment_only_omits_words():
    api = _api(_BatchBackend(), has_batch=True)
    body = api.encode_response({"output": _result(), "params": {}, "elapsed": 0.0})
    assert "words" not in body
    assert "words" not in body["segments"][0]


def test_envelope_include_logprobs():
    import math

    api = _api(_BatchBackend(), has_batch=True)
    body = api.encode_response(
        {
            "output": _result(),
            "params": {"timestamp_granularities": ["word"], "include": ["logprobs"]},
            "elapsed": 0.0,
        }
    )
    assert body["segments"][0]["avg_logprob"] == pytest.approx(math.log(0.5))


def test_envelope_response_format_json():
    api = _api(_BatchBackend(), has_batch=True)
    body = api.encode_response(
        {"output": _result(), "params": {"response_format": "json"}, "elapsed": 0.0}
    )
    assert body == {"text": "xin chào", "task": "transcribe", "elapsed_s": 0.0}


def test_decode_request_reads_file_and_params():
    from starlette.datastructures import FormData, UploadFile

    upload = UploadFile(filename="a.wav", file=io.BytesIO(b"RIFFfake"))
    form = FormData([("file", upload), ("language", "vi"), ("timestamp_granularities[]", "word")])
    api = _api(_BatchBackend(), has_batch=True)
    item = api.decode_request(form)
    try:
        assert item["params"]["language"] == "vi"
        assert item["params"]["timestamp_granularities"] == ["word"]
        with open(item["path"], "rb") as f:
            assert f.read() == b"RIFFfake"
    finally:
        api._cleanup(item["path"])


def test_decode_request_missing_file_raises():
    from starlette.datastructures import FormData

    api = _api(_BatchBackend(), has_batch=True)
    with pytest.raises(ValueError):
        api.decode_request(FormData([("language", "vi")]))
