import pytest

from multitalker_asr.data.pipeline.align_backends import (
    ALIGN_REGISTRY,
    AlignedWord,
    AlignResult,
    BaseAlignBackend,
    Qwen3AlignBackend,
    alignment_score,
    build_align_backend,
    list_align_backends,
    register_align_backend,
)


class TestAlignResult:
    def test_span(self):
        words = [AlignedWord("a", 0.0, 0.4), AlignedWord("b", 0.4, 1.2)]
        assert AlignResult(words=words).span == (0.0, 1.2)

    def test_span_empty(self):
        assert AlignResult(words=[]).span is None

    def test_to_records(self):
        recs = AlignResult(words=[AlignedWord("x", 0.1, 0.5)]).to_records()
        assert recs == [{"text": "x", "start_time": 0.1, "end_time": 0.5}]


class TestAlignmentScore:
    def test_all_valid(self):
        words = [AlignedWord("a", 0.0, 0.3), AlignedWord("b", 0.3, 0.7)]
        assert alignment_score(words) == 1.0

    def test_partial(self):
        words = [AlignedWord("a", 0.0, 0.3), AlignedWord("b", 0.7, 0.5)]
        assert alignment_score(words) == 0.5

    def test_empty(self):
        assert alignment_score([]) == 0.0


class TestRegistry:
    def test_registered(self):
        assert {"qwen3", "qwen3_service", "nemo_nfa", "mms_fa",
                "funasr_align", "funasr_nano_align"} <= set(list_align_backends())

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown align backend"):
            build_align_backend("nope")

    def test_custom_registration(self):
        class _Stub(BaseAlignBackend):
            name = "_stub_align"

            def load(self, device="cpu"):
                self._loaded = True

            def align(self, audio_path, text, language):
                return AlignResult(words=[AlignedWord("ok", 0.0, 1.0)], score=1.0)

        register_align_backend("_stub_align", _Stub)
        try:
            backend = build_align_backend("_stub_align")
            backend.load()
            out = backend.align("a.wav", "ok", "Vietnamese")
            assert out.words[0].text == "ok"
        finally:
            ALIGN_REGISTRY.pop("_stub_align", None)


class TestQwen3Backend:
    def test_language_support(self):
        backend = Qwen3AlignBackend()
        assert backend.supports_language("English")
        assert not backend.supports_language("Vietnamese")

    def test_align_parses_items_and_scores(self):
        class _Item:
            def __init__(self, text, start, end):
                self.text = text
                self.start_time = start
                self.end_time = end

        class _FakeAligner:
            def align(self, audio, text, language):
                return [[_Item("xin", 0.0, 0.3), _Item("chao", 0.3, 0.8)]]

        backend = Qwen3AlignBackend()
        backend._aligner = _FakeAligner()
        backend._loaded = True
        out = backend.align("a.wav", "xin chao", "Vietnamese")
        assert [w.text for w in out.words] == ["xin", "chao"]
        assert out.score == 1.0
        assert out.span == (0.0, 0.8)

    def test_align_empty_result(self):
        class _FakeAligner:
            def align(self, audio, text, language):
                return [[]]

        backend = Qwen3AlignBackend()
        backend._aligner = _FakeAligner()
        backend._loaded = True
        out = backend.align("a.wav", "x", "Vietnamese")
        assert out.words == []
        assert out.score == 0.0


class TestMMSSoundfileLoad:
    def test_align_uses_soundfile_not_torchaudio_load(self, tmp_path, monkeypatch):
        import numpy as np
        import soundfile as sf
        import torchaudio
        from multitalker_asr.data.pipeline.align_backends.mms_fa import MMSAlignBackend

        monkeypatch.setattr(sf, "read", lambda *a, **k: (np.zeros(16000, dtype="float32"), 16000))

        def _boom(*a, **k):
            raise AssertionError("torchaudio.load must not be called (torchcodec missing)")
        monkeypatch.setattr(torchaudio, "load", _boom)

        import collections
        Span = collections.namedtuple("Span", ["start", "end", "score"])

        class _Emission:
            shape = (1, 8, 5)
            def __getitem__(self, i):
                return self

        be = MMSAlignBackend(device="cpu")
        be._sample_rate = 16000
        be._vocab = set("abcdefghijklmnopqrstuvwxyz'-")
        be._model = lambda wf: (_Emission(), None)
        be._tokenizer = lambda words: words
        be._aligner = lambda emission0, tokens: [[Span(0, 1, 1.0)] for _ in tokens]
        be._loaded = True

        out = be.align(str(tmp_path / "a.wav"), "xin chào bạn", "Vietnamese")
        # romanized for tokenizing (đ→d, à→a) but ORIGINAL diacritics kept in output
        assert [w.text for w in out.words] == ["xin", "chào", "bạn"]


class TestQwen3ServiceAlign:
    def test_parse_words_from_http(self, tmp_path):
        from multitalker_asr.data.pipeline.align_backends.qwen3_service import (
            Qwen3ServiceAlignBackend,
        )
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"\0" * 64)

        class _Resp:
            def raise_for_status(self):
                return None
            def json(self):
                return {"words": [
                    {"text": "xin", "start": 0.0, "end": 0.5},
                    {"text": "chào", "start": 0.5, "end": 1.0},
                ]}

        class _Session:
            def post(self, *a, **k):
                return _Resp()

        be = Qwen3ServiceAlignBackend(base_url="http://x")
        be._session = _Session()
        be._loaded = True
        out = be.align(str(wav), "xin chào", "Vietnamese")
        assert [w.text for w in out.words] == ["xin", "chào"]
        assert out.span == (0.0, 1.0)


class TestFunASRAlign:
    def test_parse_timestamps_ms_to_s_and_word_map(self):
        from multitalker_asr.data.pipeline.align_backends.funasr_align import FunASRAlignBackend

        class _Model:
            def generate(self, **k):
                return [{"text": "xin chào bạn",
                         "timestamp": [[0, 500], [500, 1000], [1000, 1600]]}]

        be = FunASRAlignBackend()
        be._model = _Model()
        be._loaded = True
        out = be.align("a.wav", "xin chào bạn", "Vietnamese")
        assert [w.text for w in out.words] == ["xin", "chào", "bạn"]
        assert out.words[0].start_time == 0.0 and out.words[0].end_time == 0.5
        assert out.words[2].end_time == 1.6

    def test_proportional_when_token_count_differs(self):
        from multitalker_asr.data.pipeline.align_backends.funasr_align import FunASRAlignBackend

        class _Model:
            def generate(self, **k):
                return [{"timestamp": [[0, 100], [100, 200], [200, 300],
                                       [300, 400], [400, 500], [500, 600]]}]

        be = FunASRAlignBackend()
        be._model = _Model()
        be._loaded = True
        out = be.align("a.wav", "a b c", "vi")          # 6 spans → 3 words
        assert len(out.words) == 3
        assert out.words[0].start_time == 0.0
        assert out.words[-1].end_time == 0.6


class TestFunASRNanoGrouping:
    def test_group_subwords_into_words_on_leading_space(self):
        from multitalker_asr.data.pipeline.align_backends.funasr_nano_align import (
            FunASRNanoAlignBackend,
        )
        items = [
            {"token": "xin", "start_time": 0.0, "end_time": 0.3},
            {"token": " ch", "start_time": 0.3, "end_time": 0.5},
            {"token": "ào", "start_time": 0.5, "end_time": 0.7},
            {"token": " bạn", "start_time": 0.7, "end_time": 1.0},
        ]
        words = FunASRNanoAlignBackend._group_to_words(items)
        assert [w.text for w in words] == ["xin", "chào", "bạn"]
        assert words[1].start_time == 0.3 and words[1].end_time == 0.7
        assert words[2].start_time == 0.7 and words[2].end_time == 1.0
