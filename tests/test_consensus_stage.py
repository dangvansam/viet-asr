import pytest

from multitalker_asr.data.pipeline.checkpoint import PipelineCheckpoint
from multitalker_asr.data.pipeline.config import PipelineConfig
from multitalker_asr.data.pipeline.stages.consensus import ConsensusStage


def _aligned(start, end, n=4):
    step = (end - start) / n
    return [
        {"text": f"w{i}", "start_time": start + i * step, "end_time": start + (i + 1) * step}
        for i in range(n)
    ]


def _config(tmp_path):
    cfg = PipelineConfig(output_dir=str(tmp_path))
    cfg.consensus.min_overlap_iou = 0.5
    cfg.consensus.min_alignment_score = 0.8
    cfg.consensus.min_asr_confidence = 0.5
    cfg.consensus.drop_on_disagreement = True
    return cfg


def _run(records, cfg, tmp_path):
    ckpt = PipelineCheckpoint(checkpoint_dir=str(tmp_path / "ckpt"))
    return ConsensusStage().run(records, cfg, ckpt)


class TestConsensusStage:
    def test_agreeing_segment_kept(self, tmp_path):
        cfg = _config(tmp_path)
        rec = {
            "id": "ok",
            "duration": 2.0,
            "alignment": _aligned(0.05, 1.95),
            "alignment_score": 0.95,
            "asr_confidence": 0.9,
        }
        out = _run([rec], cfg, tmp_path)
        assert len(out) == 1
        assert out[0]["consensus"]["coverage"] is True
        assert 0.0 < out[0]["consensus_score"] <= 1.0

    def test_low_coverage_dropped(self, tmp_path):
        cfg = _config(tmp_path)
        rec = {
            "id": "bad",
            "duration": 2.0,
            "alignment": _aligned(0.0, 0.4),
            "alignment_score": 0.95,
            "asr_confidence": 0.9,
        }
        out = _run([rec], cfg, tmp_path)
        assert out == []

    def test_low_alignment_score_dropped(self, tmp_path):
        cfg = _config(tmp_path)
        rec = {
            "id": "bad",
            "duration": 2.0,
            "alignment": _aligned(0.05, 1.95),
            "alignment_score": 0.4,
            "asr_confidence": 0.9,
        }
        out = _run([rec], cfg, tmp_path)
        assert out == []

    def test_low_asr_confidence_dropped(self, tmp_path):
        cfg = _config(tmp_path)
        rec = {
            "id": "bad",
            "duration": 2.0,
            "alignment": _aligned(0.05, 1.95),
            "alignment_score": 0.95,
            "asr_confidence": 0.1,
        }
        out = _run([rec], cfg, tmp_path)
        assert out == []

    def test_no_drop_when_disabled_records_score(self, tmp_path):
        cfg = _config(tmp_path)
        cfg.consensus.drop_on_disagreement = False
        rec = {
            "id": "bad",
            "duration": 2.0,
            "alignment": _aligned(0.0, 0.4),
            "alignment_score": 0.4,
            "asr_confidence": 0.1,
        }
        out = _run([rec], cfg, tmp_path)
        assert len(out) == 1
        assert out[0]["consensus_score"] < 0.5
        assert out[0]["extra"]["consensus"] == out[0]["consensus"]


def _good_align_record(**over):
    rec = {
        "id": "r",
        "duration": 2.0,
        "alignment": _aligned(0.05, 1.95),
        "alignment_score": 0.95,
        "asr_confidence": 0.9,
        "text": "xin chào các bạn hôm nay",
    }
    rec.update(over)
    return rec


class TestConsensusSubtitleCheck:
    def test_subtitle_agreement_pass(self, tmp_path):
        cfg = _config(tmp_path)
        cfg.consensus.require_subtitle_agreement = True
        cfg.consensus.subtitle_min_similarity = 0.6
        rec = _good_align_record(
            extra={"subtitle_text": "xin chào các bạn hôm nay"}
        )
        out = _run([rec], cfg, tmp_path)
        assert len(out) == 1
        assert out[0]["consensus"]["subtitle"] is True

    def test_subtitle_disagreement_dropped(self, tmp_path):
        cfg = _config(tmp_path)
        cfg.consensus.require_subtitle_agreement = True
        cfg.consensus.subtitle_min_similarity = 0.6
        rec = _good_align_record(
            extra={"subtitle_text": "hoàn toàn khác biệt không liên quan gì"}
        )
        out = _run([rec], cfg, tmp_path)
        assert out == []

    def test_no_subtitle_no_check(self, tmp_path):
        cfg = _config(tmp_path)
        cfg.consensus.require_subtitle_agreement = True
        rec = _good_align_record()
        out = _run([rec], cfg, tmp_path)
        assert len(out) == 1
        assert "subtitle" not in out[0]["consensus"]


class TestConsensusAttributeCheck:
    def test_attribute_agreement_pass(self, tmp_path):
        cfg = _config(tmp_path)
        cfg.consensus.attribute_check_axes = ["gender"]
        cfg.consensus.require_attribute_agreement = True
        rec = _good_align_record(gender="female", extra={"weak_labels": {"gender": "female"}})
        out = _run([rec], cfg, tmp_path)
        assert len(out) == 1
        assert out[0]["consensus"]["attr_gender"] is True

    def test_attribute_disagreement_dropped(self, tmp_path):
        cfg = _config(tmp_path)
        cfg.consensus.attribute_check_axes = ["gender"]
        cfg.consensus.require_attribute_agreement = True
        rec = _good_align_record(gender="male", extra={"weak_labels": {"gender": "female"}})
        out = _run([rec], cfg, tmp_path)
        assert out == []

    def test_attribute_soft_signal_when_not_required(self, tmp_path):
        cfg = _config(tmp_path)
        cfg.consensus.attribute_check_axes = ["gender"]
        cfg.consensus.require_attribute_agreement = False
        rec = _good_align_record(gender="male", extra={"weak_labels": {"gender": "female"}})
        out = _run([rec], cfg, tmp_path)
        assert len(out) == 1
        assert out[0]["consensus"]["attr_gender"] is False
