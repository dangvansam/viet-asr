from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from multitalker_asr.configs import (
    ChunkPreset,
    InferenceConfig,
    StreamingProfile,
    all_preset_att_contexts,
    preset_chunk_ms,
    preset_to_att_context,
)
from multitalker_asr.eval.metrics import (
    AttributeConfusion,
    AttributeMacroF1Metric,
    AttributePipelineEvaluator,
)
from multitalker_asr.inference import (
    CacheAwareValidator,
    ChunkSwitcher,
    LatencyBenchmark,
    LatencyReport,
)


class TestChunkPreset:
    def test_chunk_ms_values(self):
        assert preset_chunk_ms(ChunkPreset.CHUNK_80MS) == 80.0
        assert preset_chunk_ms(ChunkPreset.CHUNK_160MS) == 160.0
        assert preset_chunk_ms(ChunkPreset.CHUNK_320MS) == 320.0
        assert preset_chunk_ms(ChunkPreset.CHUNK_560MS) == 560.0
        assert preset_chunk_ms(ChunkPreset.CHUNK_1120MS) == 1120.0

    def test_att_context_layout(self):
        assert preset_to_att_context(ChunkPreset.CHUNK_80MS) == [56, 0]
        assert preset_to_att_context(ChunkPreset.CHUNK_160MS) == [56, 1]
        assert preset_to_att_context(ChunkPreset.CHUNK_320MS) == [56, 3]
        assert preset_to_att_context(ChunkPreset.CHUNK_560MS) == [56, 6]
        assert preset_to_att_context(ChunkPreset.CHUNK_1120MS) == [56, 13]

    def test_custom_left_context(self):
        assert preset_to_att_context(ChunkPreset.CHUNK_80MS, left_context=70) == [70, 0]

    def test_all_preset_att_contexts(self):
        operations = all_preset_att_contexts()
        assert len(operations) == 5
        presets = [p for p, _ in operations]
        assert presets == list(ChunkPreset)


class TestStreamingProfile:
    def test_defaults(self):
        profile = StreamingProfile()
        assert profile.preset == ChunkPreset.CHUNK_1120MS
        assert profile.left_context == 56
        assert profile.att_context_size == [56, 13]
        assert profile.chunk_ms == 1120.0

    def test_string_preset_coerced(self):
        profile = StreamingProfile(preset="chunk_320ms")
        assert profile.preset == ChunkPreset.CHUNK_320MS

    def test_with_preset_immutable(self):
        original = StreamingProfile(preset=ChunkPreset.CHUNK_1120MS)
        switched = original.with_preset(ChunkPreset.CHUNK_80MS)
        assert original.preset == ChunkPreset.CHUNK_1120MS
        assert switched.preset == ChunkPreset.CHUNK_80MS

    def test_with_preset_rejects_unavailable(self):
        profile = StreamingProfile(
            preset=ChunkPreset.CHUNK_1120MS,
            available_presets=[ChunkPreset.CHUNK_1120MS, ChunkPreset.CHUNK_560MS],
        )
        with pytest.raises(ValueError, match="not in available_presets"):
            profile.with_preset(ChunkPreset.CHUNK_80MS)

    def test_select_for_latency_picks_largest_under_budget(self):
        profile = StreamingProfile()
        assert profile.select_for_latency_budget(500) == ChunkPreset.CHUNK_320MS
        assert profile.select_for_latency_budget(1200) == ChunkPreset.CHUNK_1120MS
        assert profile.select_for_latency_budget(160) == ChunkPreset.CHUNK_160MS

    def test_select_for_latency_below_minimum_returns_smallest(self):
        profile = StreamingProfile()
        assert profile.select_for_latency_budget(50) == ChunkPreset.CHUNK_80MS

    def test_invalid_left_context(self):
        with pytest.raises(ValueError, match="left_context"):
            StreamingProfile(left_context=-1)


class TestInferenceConfigDefaults:
    def test_default_att_context_is_56_13(self):
        cfg = InferenceConfig()
        assert cfg.att_context_size == [56, 13]

    def test_with_preset_helper(self):
        cfg = InferenceConfig.with_preset(ChunkPreset.CHUNK_320MS)
        assert cfg.att_context_size == [56, 3]
        assert cfg.streaming_profile is not None
        assert cfg.streaming_profile.preset == ChunkPreset.CHUNK_320MS

    def test_streaming_profile_overrides_default(self):
        profile = StreamingProfile(preset=ChunkPreset.CHUNK_80MS)
        cfg = InferenceConfig(streaming_profile=profile)
        assert cfg.att_context_size == [56, 0]


class TestCacheAwareValidator:
    def test_detect_cache_aware_present(self):
        encoder = SimpleNamespace(
            streaming_cfg=SimpleNamespace(chunk_size=14, drop_extra_pre_encoded=2),
            cache_last_channel=None,
            cache_last_time=None,
        )
        encoder.set_default_att_context_size = lambda **kw: None
        model = SimpleNamespace(encoder=encoder)
        capability = CacheAwareValidator().inspect(model)
        assert capability.is_cache_aware
        assert capability.chunk_size == 14
        assert capability.cache_drop_size == 2

    def test_detect_cache_aware_absent(self):
        model = SimpleNamespace(encoder=SimpleNamespace())
        capability = CacheAwareValidator().inspect(model)
        assert not capability.is_cache_aware
        assert "NOT cache-aware" in capability.summary()

    def test_handles_missing_encoder(self):
        capability = CacheAwareValidator().inspect(SimpleNamespace())
        assert not capability.is_cache_aware


class TestChunkSwitcher:
    def _make_model(self):
        encoder = SimpleNamespace(streaming_cfg=SimpleNamespace(chunk_size=14))
        encoder.set_default_att_context_size = MagicMock()
        return SimpleNamespace(encoder=encoder)

    def test_apply_sets_att_context(self):
        model = self._make_model()
        switcher = ChunkSwitcher(model)
        result = switcher.apply(ChunkPreset.CHUNK_320MS)
        assert result == [56, 3]
        model.encoder.set_default_att_context_size.assert_called_once_with(
            att_context_size=[56, 3]
        )
        assert switcher.current_preset == ChunkPreset.CHUNK_320MS

    def test_apply_rejects_unavailable(self):
        model = self._make_model()
        profile = StreamingProfile(
            preset=ChunkPreset.CHUNK_1120MS,
            available_presets=[ChunkPreset.CHUNK_1120MS],
        )
        switcher = ChunkSwitcher(model, profile=profile)
        with pytest.raises(ValueError, match="not in available presets"):
            switcher.apply(ChunkPreset.CHUNK_80MS)

    def test_apply_for_latency(self):
        model = self._make_model()
        switcher = ChunkSwitcher(model)
        preset = switcher.apply_for_latency(500)
        assert preset == ChunkPreset.CHUNK_320MS
        assert switcher.current_preset == ChunkPreset.CHUNK_320MS

    def test_enumerate(self):
        model = self._make_model()
        switcher = ChunkSwitcher(model)
        points = switcher.enumerate_operating_points()
        assert len(points) == 5
        assert points[0][1] == [56, 0]


class TestLatencyBenchmark:
    def test_run_collects_per_preset(self):
        model = SimpleNamespace(
            encoder=SimpleNamespace(set_default_att_context_size=MagicMock())
        )
        switcher = ChunkSwitcher(model)
        call_count = {"n": 0}

        def fake_transcribe():
            call_count["n"] += 1

        bench = LatencyBenchmark(
            switcher,
            transcribe_fn=fake_transcribe,
            audio_seconds=10.0,
            warmup_runs=0,
            measurement_runs=2,
        )
        report = bench.run(presets=[ChunkPreset.CHUNK_320MS, ChunkPreset.CHUNK_1120MS])
        assert len(report.measurements) == 2
        assert call_count["n"] == 4
        assert all(m.real_time_factor >= 0 for m in report.measurements)
        assert report.median_rtf() is not None

    def test_report_as_dict(self):
        model = SimpleNamespace(
            encoder=SimpleNamespace(set_default_att_context_size=MagicMock())
        )
        switcher = ChunkSwitcher(model)
        bench = LatencyBenchmark(
            switcher,
            transcribe_fn=lambda: None,
            audio_seconds=5.0,
            warmup_runs=0,
            measurement_runs=1,
        )
        report = bench.run(presets=[ChunkPreset.CHUNK_80MS])
        as_dict = report.as_dict()
        assert as_dict[0]["preset"] == "chunk_80ms"
        assert as_dict[0]["chunk_ms"] == 80.0

    def test_invalid_audio_seconds(self):
        with pytest.raises(ValueError, match="audio_seconds"):
            LatencyBenchmark(
                switcher=None,
                transcribe_fn=lambda: None,
                audio_seconds=0.0,
            )


class TestAttributeConfusion:
    def test_perfect_predictions(self):
        confusion = AttributeConfusion(axis="emotion", label_space=["happy", "sad"])
        confusion.update("happy", "happy")
        confusion.update("sad", "sad")
        assert confusion.accuracy == 1.0
        assert confusion.macro_f1() == 1.0

    def test_partial_predictions(self):
        confusion = AttributeConfusion(axis="gender", label_space=["male", "female"])
        confusion.update("male", "male")
        confusion.update("male", "female")
        confusion.update("female", "female")
        confusion.update("female", "female")
        assert confusion.accuracy == 0.75
        f1 = confusion.per_label_f1()
        assert f1["female"] > f1["male"]

    def test_precision_recall(self):
        confusion = AttributeConfusion(axis="emotion", label_space=["happy", "sad"])
        confusion.update("happy", "happy")
        confusion.update("happy", "happy")
        confusion.update("sad", "happy")
        pr = confusion.precision_recall()
        assert pr["happy"]["precision"] == pytest.approx(2 / 3)
        assert pr["happy"]["recall"] == 1.0
        assert pr["sad"]["recall"] == 0.0


class TestAttributeMacroF1Metric:
    def test_compute_returns_recorded_marker(self):
        metric = AttributeMacroF1Metric(label_space=["happy", "sad"], axis="emotion")
        out = metric.compute("happy", "happy")
        assert out == {"emotion_recorded": 1.0}

    def test_compute_handles_missing(self):
        metric = AttributeMacroF1Metric(label_space=["happy", "sad"], axis="emotion")
        out = metric.compute(None, "happy")
        assert out == {"emotion_recorded": 0.0}

    def test_aggregate_uses_internal_confusion(self):
        metric = AttributeMacroF1Metric(label_space=["happy", "sad"], axis="emotion")
        metric.compute("happy", "happy")
        metric.compute("sad", "happy")
        agg = metric.aggregate([])
        assert agg["emotion_accuracy"] == 0.5
        assert agg["emotion_samples"] == 2

    def test_reset(self):
        metric = AttributeMacroF1Metric(label_space=["happy", "sad"], axis="emotion")
        metric.compute("happy", "happy")
        metric.reset()
        assert metric.confusion.total == 0


class TestAttributePipelineEvaluator:
    def test_full_5_axes(self):
        axes = {
            "language": ["vi-VN", "en-US"],
            "emotion": ["happy", "sad", "neutral"],
            "gender": ["male", "female"],
            "age": ["child", "adult"],
            "region": ["northern", "central", "southern"],
        }
        evaluator = AttributePipelineEvaluator(axes)
        evaluator.update(
            references={
                "language": "vi-VN",
                "emotion": "happy",
                "gender": "male",
                "age": "adult",
                "region": "southern",
            },
            predictions={
                "language": "vi-VN",
                "emotion": "happy",
                "gender": "female",
                "age": "adult",
                "region": "central",
            },
        )
        report = evaluator.report()
        assert report["language"]["language_accuracy"] == 1.0
        assert report["gender"]["gender_accuracy"] == 0.0
        assert "per_label_f1" in report["emotion"]

    def test_average_macro_f1(self):
        evaluator = AttributePipelineEvaluator(
            {"emotion": ["happy", "sad"], "gender": ["male", "female"]}
        )
        evaluator.update({"emotion": "happy", "gender": "male"}, {"emotion": "happy", "gender": "male"})
        evaluator.update({"emotion": "sad", "gender": "female"}, {"emotion": "sad", "gender": "female"})
        assert evaluator.average_macro_f1() == 1.0

    def test_update_batch(self):
        evaluator = AttributePipelineEvaluator({"emotion": ["happy", "sad"]})
        pairs = [
            ({"emotion": "happy"}, {"emotion": "happy"}),
            ({"emotion": "sad"}, {"emotion": "happy"}),
        ]
        evaluator.update_batch(pairs)
        report = evaluator.report()
        assert report["emotion"]["emotion_samples"] == 2
        assert report["emotion"]["emotion_accuracy"] == 0.5

    def test_reset(self):
        evaluator = AttributePipelineEvaluator({"emotion": ["happy", "sad"]})
        evaluator.update({"emotion": "happy"}, {"emotion": "happy"})
        evaluator.reset()
        report = evaluator.report()
        assert report["emotion"]["emotion_samples"] == 0
