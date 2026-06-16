import numpy as np
import pytest

from multitalker_asr.configs.transcript_pipeline import (
    ASRBackendConfig,
    TranscriptPipelineConfig,
)
from multitalker_asr.data.pipeline.asr_backends import (
    ASRResult,
    WordTiming,
    WordRoverEnsembler,
    register_asr_backend,
)
from multitalker_asr.data.pipeline.asr_backends.base import BaseASRBackend
from multitalker_asr.data.pipeline.checkpoint import PipelineCheckpoint
from multitalker_asr.data.pipeline.config import PipelineConfig
from multitalker_asr.data.pipeline.stages.transcript_refine import TranscriptRefineStage


def _wt(pairs):
    return [WordTiming(word=w, start=float(i), end=float(i + 1)) for i, w in enumerate(pairs)]


class TestWordRover:
    def test_corrects_wrong_anchor_word_when_backend_and_subtitle_agree(self):
        anchor = ASRResult(text="xin chao2 các bạn", word_timings=_wt(["xin", "chao2", "các", "bạn"]), backend="primary")
        backend = ASRResult(text="xin chào các bạn", word_timings=_wt(["xin", "chào", "các", "bạn"]), backend="funasr_mlt")
        subtitle = ASRResult(text="xin chào các bạn", word_timings=_wt(["xin", "chào", "các", "bạn"]), backend="subtitle")
        fused = WordRoverEnsembler(subtitle_weight=1.5).combine([anchor, backend], reference=subtitle)
        assert fused.text == "xin chào các bạn"            # wrong word corrected
        assert fused.raw["words_changed"] == 1

    def test_preserves_anchor_punctuation_on_agreement(self):
        anchor = ASRResult(text="xin chào bạn.", word_timings=_wt(["xin", "chào", "bạn."]), backend="primary")
        backend = ASRResult(text="xin chào bạn", word_timings=_wt(["xin", "chào", "bạn"]), backend="funasr_mlt")
        subtitle = ASRResult(text="xin chào bạn", word_timings=_wt(["xin", "chào", "bạn"]), backend="subtitle")
        fused = WordRoverEnsembler().combine([anchor, backend], reference=subtitle)
        assert fused.text == "xin chào bạn."               # punctuation kept (anchor wins ties)
        assert fused.raw["words_changed"] == 0

    def test_text_fallback_picks_closest_to_subtitle(self):
        a = ASRResult(text="hello world", backend="primary")          # no timings
        b = ASRResult(text="xin chào các bạn", backend="vietasr")
        subtitle = ASRResult(text="xin chao cac ban", backend="subtitle")  # no timings
        fused = WordRoverEnsembler().combine([a, b], reference=subtitle)
        assert fused.text == "xin chào các bạn"            # closest to subtitle wins

    def test_no_reference_keeps_anchor(self):
        a = ASRResult(text="anchor text", word_timings=_wt(["anchor", "text"]), backend="primary")
        fused = WordRoverEnsembler().combine([a], reference=None)
        assert fused.text == "anchor text"

    def test_vote_weight_overrides_default(self):
        ens = WordRoverEnsembler()
        assert ens._weight(ASRResult(text="x", backend="primary", vote_weight=0.5)) == 0.5
        assert ens._weight(ASRResult(text="x", backend="qwen3")) == 1.0
        assert ens._weight(ASRResult(text="x", backend="subtitle")) == 1.5

    def test_nbest_candidates_soft_vote_dont_outvote_full_backend(self):
        # funasr n-best (3 candidates @ 1/3 each) all keep the wrong anchor word,
        # but one full-weight backend + subtitle agree on the correct word → corrected.
        w = 1.0 / 3
        anchor = ASRResult(text="xin chao2 bạn", word_timings=_wt(["xin", "chao2", "bạn"]), backend="primary", vote_weight=w)
        alt1 = ASRResult(text="xin chao2 bạn", word_timings=_wt(["xin", "chao2", "bạn"]), backend="primary_alt1", vote_weight=w)
        alt2 = ASRResult(text="xin chao2 bạn", word_timings=_wt(["xin", "chao2", "bạn"]), backend="primary_alt2", vote_weight=w)
        backend = ASRResult(text="xin chào bạn", word_timings=_wt(["xin", "chào", "bạn"]), backend="qwen3")
        subtitle = ASRResult(text="xin chào bạn", word_timings=_wt(["xin", "chào", "bạn"]), backend="subtitle")
        fused = WordRoverEnsembler().combine([anchor, alt1, alt2, backend], reference=subtitle)
        assert fused.text == "xin chào bạn"


class _StubBackend(BaseASRBackend):
    name = "stub_refine"
    languages = ["vi"]

    def __init__(self, **kwargs):
        self._loaded = False

    def load(self, device="cpu"):
        self._loaded = True

    def transcribe(self, audio, sample_rate, language=None):
        return ASRResult(
            text="xin chào các bạn",
            word_timings=_wt(["xin", "chào", "các", "bạn"]),
            backend=self.name,
        )


class _TextOnlyBackend(BaseASRBackend):
    """Returns text WITHOUT word timings (like vietasr/qwen3)."""
    name = "stub_textonly"
    languages = ["vi"]

    def __init__(self, **kwargs):
        self._loaded = False

    def load(self, device="cpu"):
        self._loaded = True

    def transcribe(self, audio, sample_rate, language=None):
        return ASRResult(text="xin chào các bạn", word_timings=None, backend=self.name)


class _StubAligner:
    """Forced aligner that returns one word span per token (0-based, 1s each)."""
    name = "stub_align"

    def __init__(self, **kwargs):
        self._loaded = False

    def load(self, device="cpu"):
        self._loaded = True

    def align(self, audio_path, text, language):
        from multitalker_asr.data.pipeline.align_backends import AlignResult, AlignedWord
        toks = text.split()
        return AlignResult(
            words=[AlignedWord(t, float(i), float(i + 1)) for i, t in enumerate(toks)],
            score=1.0, backend=self.name,
        )

    def unload(self):
        self._loaded = False


class _StubLoader:
    def load(self, path, offset=0.0, duration=None):
        return np.zeros(16000, dtype=np.float32), 16000


register_asr_backend("stub_refine", _StubBackend)
register_asr_backend("stub_textonly", _TextOnlyBackend)


class TestTranscriptRefineStage:
    def _stage(self, use_subtitle=True):
        cfg = TranscriptPipelineConfig(
            backends=[ASRBackendConfig(name="stub_refine", device="cpu")],
            ensemble_strategy="rover",
            use_subtitle=use_subtitle,
            align_backend="none",            # no forced aligner for this test
        )
        return TranscriptRefineStage(cfg, audio_loader=_StubLoader())

    def test_corrects_text_preserves_raw_and_sets_provenance(self, tmp_path):
        stage = self._stage()
        vtt = tmp_path / "s.vtt"
        vtt.write_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:04.000\nxin chào các bạn\n",
            encoding="utf-8",
        )
        rec = {
            "id": "seg1",
            "audio_filepath": str(tmp_path / "a.wav"),
            "text": "xin chao2 các bạn",
            "text_raw": "xin chao2 cac ban",
            "start": 0.0,
            "end": 4.0,
            "extra": {"subtitle_path": str(vtt)},
            "alignment": [
                {"text": w, "start_time": float(i), "end_time": float(i + 1)}
                for i, w in enumerate(["xin", "chao2", "các", "bạn"])
            ],
        }
        ck = PipelineCheckpoint(checkpoint_dir=str(tmp_path / "ck"))
        out = stage.run([rec], PipelineConfig(output_dir=str(tmp_path)), ck)
        r = out[0]
        assert r["text"] == "xin chào các bạn"             # backend + subtitle corrected it
        assert r["text_itn"] == "xin chào các bạn"
        assert r["text_raw"] == "xin chao2 cac ban"         # raw untouched
        assert r["extra"]["asr_chosen"] == "rover"
        assert r["extra"]["subtitle_used"] is True
        assert "stub_refine" in r["extra"]["asr_candidates"]
        assert len(r["alignment"]) == 4                      # fused word timings kept

    def test_subtitle_window_reference(self, tmp_path):
        vtt = tmp_path / "s.vtt"
        vtt.write_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:04.000\nxin chào các bạn\n",
            encoding="utf-8",
        )
        from multitalker_asr.utils.subtitle import VTTParser
        words = VTTParser().words(str(vtt), 0.0, 4.0)
        assert [w.word for w in words] == ["xin", "chào", "các", "bạn"]
        assert words[0].start == pytest.approx(0.0)
        assert words[-1].end == pytest.approx(4.0)

    def test_build_anchor_group_expands_nbest_with_inverse_weight(self):
        stage = self._stage()
        rec = {
            "text": "xin chào bạn",
            "alignment": [
                {"text": w, "start_time": float(i), "end_time": float(i + 1)}
                for i, w in enumerate(["xin", "chào", "bạn"])
            ],
            "extra": {"asr_candidates_nbest": ["xin chào bạn", "xin chao bạn", "xin chào ban"]},
        }
        group = stage._build_anchor_group(rec, "xin chào bạn", "vi")
        assert [h.backend for h in group] == ["primary", "primary_alt1", "primary_alt2"]
        assert all(abs(h.vote_weight - 1.0 / 3) < 1e-9 for h in group)
        assert group[0].word_timings is not None and group[1].word_timings is None

    def test_build_anchor_group_single_when_no_candidates(self):
        stage = self._stage()
        group = stage._build_anchor_group({"text": "xin chào bạn"}, "xin chào bạn", "vi")
        assert len(group) == 1
        assert group[0].vote_weight is None

    def test_forced_align_lets_textonly_backend_vote(self, tmp_path):
        from multitalker_asr.data.pipeline.align_backends import register_align_backend
        register_align_backend("stub_align", _StubAligner)
        cfg = TranscriptPipelineConfig(
            backends=[ASRBackendConfig(name="stub_textonly", device="cpu")],
            ensemble_strategy="rover",
            use_subtitle=False,
            align_backend="stub_align",       # force-aligns the text-only hypothesis
            align_device="cpu",
        )
        stage = TranscriptRefineStage(cfg, audio_loader=_StubLoader())
        # Anchor has a wrong word; the text-only backend (after forced alignment)
        # plus subtitle-less majority must still NOT change it on a tie — but with
        # the aligned backend agreeing twice it overrides. Use 2 text-only voters.
        cfg.backends.append(ASRBackendConfig(name="stub_textonly", device="cpu"))
        rec = {
            "id": "seg1",
            "audio_filepath": str(tmp_path / "a.wav"),
            "text": "xin chao2 các bạn",
            "alignment": [
                {"text": w, "start_time": float(i), "end_time": float(i + 1)}
                for i, w in enumerate(["xin", "chao2", "các", "bạn"])
            ],
        }
        ck = PipelineCheckpoint(checkpoint_dir=str(tmp_path / "ck"))
        out = stage.run([rec], PipelineConfig(output_dir=str(tmp_path)), ck)
        r = out[0]
        assert r["text"] == "xin chào các bạn"                 # corrected via aligned votes
        assert "stub_textonly" in r["extra"].get("asr_force_aligned", [])
