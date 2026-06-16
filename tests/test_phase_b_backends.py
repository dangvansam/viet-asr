import numpy as np
import pytest

from multitalker_asr.configs import (
    ASRBackendConfig,
    AttributeAxisConfig,
    AttributeBackendConfig,
    AttributePipelineConfig,
    ITNBackendConfig,
    ITNPipelineConfig,
    LLMConfig,
    TranscriptPipelineConfig,
)
from multitalker_asr.data.pipeline.asr_backends import (
    ASR_REGISTRY,
    ASREnsemblerFactory,
    ASRResult,
    BaseASRBackend,
    SingleASREnsembler,
    VoteASREnsembler,
    build_asr_backend,
    list_asr_backends,
    register_asr_backend,
)
from multitalker_asr.data.pipeline.attribute_backends import (
    ATTRIBUTE_REGISTRY,
    AttributeEnsembler,
    AttributeResult,
    BaseAttributeBackend,
)
from multitalker_asr.data.pipeline.attribute_backends.language import (
    TagDerivedLanguageBackend,
)
from multitalker_asr.data.pipeline.itn_backends import (
    ITN_REGISTRY,
    BaseITNBackend,
    FirstSuccessStrategy,
    ITNResult,
    ITNStrategyFactory,
    VoteITNStrategy,
)
from multitalker_asr.data.pipeline.stages.filter import (
    AttributeConfidenceRule,
    DurationRatioRule,
    FilterStage,
    TextLengthRule,
)


class _FakeASRBackend(BaseASRBackend):
    def __init__(self, name, text, confidence=1.0, language=None):
        self.name = name
        self.languages = []
        self._text = text
        self._confidence = confidence
        self._language = language
        self._loaded = False

    def load(self, device="cpu"):
        self._loaded = True

    def transcribe(self, audio, sample_rate, language=None):
        return ASRResult(
            text=self._text,
            confidence=self._confidence,
            language=self._language or language,
            backend=self.name,
        )


class _FakeITNBackend(BaseITNBackend):
    def __init__(self, name, normalized, raise_on_call=False):
        self.name = name
        self.languages = ["vi"]
        self._normalized = normalized
        self._raise = raise_on_call
        self._loaded = False

    def load(self):
        self._loaded = True

    def normalize(self, text, language="vi"):
        if self._raise:
            raise RuntimeError("simulated failure")
        return ITNResult(text_itn=self._normalized, backend=self.name, language=language)


class _FakeAttributeBackend(BaseAttributeBackend):
    def __init__(self, axis, name, label, confidence=0.8, posterior=None):
        self.axis = axis
        self.name = name
        self.label_space = []
        self._label = label
        self._confidence = confidence
        self._posterior = posterior
        self._loaded = False

    def load(self, device="cpu"):
        self._loaded = True

    def predict(self, audio, sample_rate, text=None):
        return AttributeResult(
            axis=self.axis,
            label=self._label,
            confidence=self._confidence,
            posterior=self._posterior,
            backend=self.name,
        )


class TestASRRegistry:
    def test_registered_backends(self):
        names = list_asr_backends()
        assert "funasr" in names
        assert "nemotron" in names
        assert "vietasr" in names
        assert "openai_transcription" in names

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown ASR backend"):
            build_asr_backend("does_not_exist")

    def test_custom_backend_registration(self):
        class _Stub(BaseASRBackend):
            name = "_stub"
            languages = ["vi"]

            def load(self, device="cpu"):
                self._loaded = True

            def transcribe(self, audio, sample_rate, language=None):
                return ASRResult(text="ok", backend=self.name)

        register_asr_backend("_stub", _Stub)
        try:
            backend = build_asr_backend("_stub")
            backend.load()
            result = backend.transcribe(np.zeros(16000, np.float32), 16000)
            assert result.text == "ok"
        finally:
            ASR_REGISTRY.pop("_stub", None)


class TestASREnsemblers:
    def _make_hyps(self):
        return [
            ASRResult(text="xin chào", confidence=0.9, backend="a"),
            ASRResult(text="xin chao", confidence=0.85, backend="b"),
            ASRResult(text="xin chào", confidence=0.95, backend="c"),
        ]

    def test_single_returns_first(self):
        ens = SingleASREnsembler()
        out = ens.combine(self._make_hyps())
        assert out.text == "xin chào"
        assert out.backend == "a"

    def test_single_empty(self):
        ens = SingleASREnsembler()
        out = ens.combine([])
        assert out.text == ""
        assert out.confidence == 0.0

    def test_vote_picks_largest_cluster(self):
        ens = VoteASREnsembler(similarity_threshold=0.7)
        out = ens.combine(self._make_hyps())
        assert out.text == "xin chào"
        assert out.raw["agree"] >= 2

    def test_vote_with_single_hyp(self):
        ens = VoteASREnsembler()
        out = ens.combine([ASRResult(text="only", confidence=0.5, backend="a")])
        assert out.text == "only"

    def test_factory_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown ensemble strategy"):
            ASREnsemblerFactory.build("not_a_strategy")

    def test_factory_single(self):
        ens = ASREnsemblerFactory.build("single")
        assert isinstance(ens, SingleASREnsembler)

    def test_factory_vote(self):
        ens = ASREnsemblerFactory.build("vote", min_agree=2)
        assert isinstance(ens, VoteASREnsembler)

    def test_factory_llm_judge_requires_client(self):
        with pytest.raises(ValueError, match="requires llm_client"):
            ASREnsemblerFactory.build("llm_judge")


class TestITNStrategies:
    def test_first_success_returns_first(self):
        backends = [
            _FakeITNBackend("primary", "Result."),
            _FakeITNBackend("fallback", "Fallback."),
        ]
        for b in backends:
            b.load()
        strategy = FirstSuccessStrategy()
        out = strategy.apply(backends, "result", "vi")
        assert out.text_itn == "Result."
        assert out.backend == "primary"

    def test_first_success_fallback_on_failure(self):
        backends = [
            _FakeITNBackend("primary", "x", raise_on_call=True),
            _FakeITNBackend("fallback", "fallback ok"),
        ]
        for b in backends:
            b.load()
        out = FirstSuccessStrategy().apply(backends, "hi", "vi")
        assert out.text_itn == "fallback ok"
        assert out.backend == "fallback"

    def test_first_success_empty_text(self):
        out = FirstSuccessStrategy().apply([], "", "vi")
        assert out.text_itn == ""

    def test_vote_strategy_majority(self):
        backends = [
            _FakeITNBackend("a", "Same."),
            _FakeITNBackend("b", "Same."),
            _FakeITNBackend("c", "Different."),
        ]
        for b in backends:
            b.load()
        out = VoteITNStrategy().apply(backends, "same", "vi")
        assert out.text_itn == "Same."

    def test_factory_first_success(self):
        s = ITNStrategyFactory.build("first_success")
        assert isinstance(s, FirstSuccessStrategy)

    def test_factory_unknown(self):
        with pytest.raises(ValueError, match="Unknown ITN strategy"):
            ITNStrategyFactory.build("nope")


class TestITNRegistry:
    def test_registered_backends(self):
        assert "llm_itn" in ITN_REGISTRY
        assert "funasr_itn" in ITN_REGISTRY
        assert "vietasr_itn" in ITN_REGISTRY
        assert "nemo_itn" in ITN_REGISTRY


class TestAttributeEnsembler:
    def test_vote_majority(self):
        backends = [
            _FakeAttributeBackend("emotion", "a", "happy", 0.8),
            _FakeAttributeBackend("emotion", "b", "happy", 0.9),
            _FakeAttributeBackend("emotion", "c", "sad", 0.7),
        ]
        ens = AttributeEnsembler(backends, axis="emotion", strategy="vote")
        out = ens.predict(np.zeros(16000, np.float32), 16000)
        assert out.label == "happy"
        assert out.raw["votes"]["happy"] == 2

    def test_first_strategy(self):
        backends = [
            _FakeAttributeBackend("gender", "a", "male", 0.9),
            _FakeAttributeBackend("gender", "b", "female", 0.6),
        ]
        ens = AttributeEnsembler(backends, axis="gender", strategy="first")
        out = ens.predict(np.zeros(16000, np.float32), 16000)
        assert out.label == "male"

    def test_mean_posterior(self):
        backends = [
            _FakeAttributeBackend(
                "age", "a", "adult", 0.7, posterior={"adult": 0.7, "senior": 0.3}
            ),
            _FakeAttributeBackend(
                "age", "b", "senior", 0.8, posterior={"adult": 0.4, "senior": 0.6}
            ),
        ]
        ens = AttributeEnsembler(backends, axis="age", strategy="mean_posterior")
        out = ens.predict(np.zeros(16000, np.float32), 16000)
        assert out.label in ("adult", "senior")
        assert out.posterior is not None
        assert pytest.approx(sum(out.posterior.values()), rel=1e-6) == 1.0

    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown attribute strategy"):
            AttributeEnsembler(
                [_FakeAttributeBackend("emotion", "a", "happy")],
                axis="emotion",
                strategy="nope",
            )

    def test_empty_backends(self):
        with pytest.raises(ValueError, match="at least one backend"):
            AttributeEnsembler([], axis="emotion")


class TestAttributeRegistry:
    def test_axes(self):
        assert "language" in ATTRIBUTE_REGISTRY
        assert "emotion" in ATTRIBUTE_REGISTRY
        assert "gender" in ATTRIBUTE_REGISTRY
        assert "age" in ATTRIBUTE_REGISTRY
        assert "region" in ATTRIBUTE_REGISTRY


class TestTagDerivedLanguage:
    def test_parses_locale_tag(self):
        backend = TagDerivedLanguageBackend()
        backend.load()
        out = backend.predict(np.zeros(16000, np.float32), 16000, text="hello <en-US>")
        assert out.label == "en-US"

    def test_fallback(self):
        backend = TagDerivedLanguageBackend(fallback="vi-VN")
        backend.load()
        out = backend.predict(np.zeros(16000, np.float32), 16000, text="no tag here")
        assert out.label == "vi-VN"


class TestPipelineConfigs:
    def test_transcript_pipeline_defaults(self):
        cfg = TranscriptPipelineConfig()
        assert cfg.ensemble_strategy == "single"
        assert len(cfg.backends) == 1
        assert cfg.backends[0].name == "funasr"

    def test_transcript_pipeline_llm_judge_requires_llm(self):
        with pytest.raises(ValueError, match="requires llm_judge"):
            TranscriptPipelineConfig(ensemble_strategy="llm_judge")

    def test_transcript_pipeline_empty_backends(self):
        with pytest.raises(ValueError, match="must be non-empty"):
            TranscriptPipelineConfig(backends=[])

    def test_transcript_pipeline_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown ensemble_strategy"):
            TranscriptPipelineConfig(ensemble_strategy="unknown")

    def test_transcript_pipeline_multi_backend(self):
        cfg = TranscriptPipelineConfig(
            backends=[
                ASRBackendConfig(name="nemotron", weight=1.0),
                ASRBackendConfig(name="vietasr", weight=0.8),
                ASRBackendConfig(name="funasr", weight=0.6, enabled=False),
            ],
            ensemble_strategy="vote",
            min_agree=2,
        )
        assert len(cfg.backends) == 3
        assert len(cfg.enabled_backends()) == 2

    def test_itn_pipeline_defaults(self):
        cfg = ITNPipelineConfig()
        assert cfg.strategy == "first_success"

    def test_itn_pipeline_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            ITNPipelineConfig(strategy="nope")

    def test_itn_pipeline_llm_judge_requires_llm(self):
        with pytest.raises(ValueError, match="requires llm"):
            ITNPipelineConfig(strategy="llm_judge")

    def test_attribute_pipeline_default_full(self):
        cfg = AttributePipelineConfig.default_full()
        axes = [a.axis for a in cfg.axes]
        assert axes == ["language", "emotion", "gender", "age", "region"]
        for axis_cfg in cfg.axes:
            assert axis_cfg.enabled_backends()

    def test_attribute_pipeline_duplicate_axis(self):
        with pytest.raises(ValueError, match="Duplicate axis"):
            AttributePipelineConfig(
                axes=[
                    AttributeAxisConfig(
                        axis="emotion",
                        backends=[AttributeBackendConfig(name="a")],
                    ),
                    AttributeAxisConfig(
                        axis="emotion",
                        backends=[AttributeBackendConfig(name="b")],
                    ),
                ]
            )

    def test_attribute_axis_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            AttributeAxisConfig(axis="emotion", strategy="nope")


class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig()
        assert cfg.provider == "openai"
        assert cfg.litellm_model == "gpt-4o-mini"

    def test_provider_prefix(self):
        cfg = LLMConfig(provider="qwen", model="qwen3-max")
        assert cfg.litellm_model == "qwen/qwen3-max"

    def test_model_with_slash_is_kept(self):
        cfg = LLMConfig(provider="qwen", model="openrouter/qwen-72b")
        assert cfg.litellm_model == "openrouter/qwen-72b"

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            LLMConfig(temperature=-0.1)

    def test_invalid_retries(self):
        with pytest.raises(ValueError, match="max_retries"):
            LLMConfig(max_retries=-1)


class TestFilterRules:
    def test_text_length_too_short(self):
        rule = TextLengthRule(min_chars=5)
        ok, reason = rule.check({"text": "hi"})
        assert not ok
        assert "shorter" in reason

    def test_text_length_too_long(self):
        rule = TextLengthRule(max_chars=10)
        ok, _ = rule.check({"text": "x" * 20})
        assert not ok

    def test_duration_ratio(self):
        rule = DurationRatioRule(min_chars_per_second=2.0, max_chars_per_second=10.0)
        ok, _ = rule.check({"text": "abc", "duration": 10.0})
        assert not ok

    def test_attribute_confidence_drop(self):
        rule = AttributeConfidenceRule(
            min_confidence={"emotion": 0.5}, drop_record=True
        )
        ok, _ = rule.check(
            {"emotion": "happy", "attribute_confidence": {"emotion": 0.3}}
        )
        assert not ok

    def test_attribute_confidence_mask(self):
        rule = AttributeConfidenceRule(
            min_confidence={"emotion": 0.5}, drop_record=False
        )
        record = {"emotion": "happy", "attribute_confidence": {"emotion": 0.3}}
        ok, _ = rule.check(record)
        assert ok
        assert record["emotion"] is None


class TestFilterStage:
    def test_keep_and_drop(self, tmp_path):
        from multitalker_asr.data.pipeline.checkpoint import PipelineCheckpoint

        ckpt = PipelineCheckpoint(checkpoint_dir=str(tmp_path / "ckpt"))
        stage = FilterStage(rules=[TextLengthRule(min_chars=3, max_chars=100)])
        records = [
            {"id": "1", "text": "hello"},
            {"id": "2", "text": "x"},
            {"id": "3", "text": "world"},
        ]
        out = stage.run(records, config=None, checkpoint=ckpt)
        assert len(out) == 2
        assert stage.stats["kept"] == 2
        assert stage.stats["dropped"] == 1
