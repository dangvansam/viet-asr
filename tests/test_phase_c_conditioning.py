import pytest
import torch

from multitalker_asr.configs import (
    ConditionerFactory,
    ConditioningConfig,
    ConditioningStrategy,
    MultiTaskConfig,
)
from multitalker_asr.models.conditioning import (
    AttributeEmbedding,
    AuxLossScheduler,
    BaseConditioner,
    DecoderTagOnlyConditioner,
    FeatureConcatConditioner,
    HybridConditioner,
    PrependPromptCEConditioner,
)
from multitalker_asr.models.prompt_embedding import TaskTokenRegistry


def _registry_full() -> TaskTokenRegistry:
    cfg = MultiTaskConfig.full_attributes()
    return TaskTokenRegistry(cfg, prefer_legacy_labels=False)


def _registry_legacy() -> TaskTokenRegistry:
    cfg = MultiTaskConfig()
    return TaskTokenRegistry(cfg, prefer_legacy_labels=True)


class TestFactoryRegistration:
    def test_all_four_strategies_registered(self):
        registered = ConditionerFactory.registered_strategies()
        assert ConditioningStrategy.PREPEND_PROMPT_CE in registered
        assert ConditioningStrategy.DECODER_TAG_ONLY in registered
        assert ConditioningStrategy.FEATURE_CONCAT in registered
        assert ConditioningStrategy.HYBRID in registered

    def test_factory_returns_correct_class(self):
        registry = _registry_full()
        mt_cfg = MultiTaskConfig.full_attributes()

        for strategy, cls in (
            (ConditioningStrategy.PREPEND_PROMPT_CE, PrependPromptCEConditioner),
            (ConditioningStrategy.DECODER_TAG_ONLY, DecoderTagOnlyConditioner),
            (ConditioningStrategy.FEATURE_CONCAT, FeatureConcatConditioner),
            (ConditioningStrategy.HYBRID, HybridConditioner),
        ):
            cfg = ConditioningConfig(strategy=strategy)
            built = ConditionerFactory.build(
                cfg, multitask_config=mt_cfg, registry=registry
            )
            assert isinstance(built, cls)


class TestAttributeEmbedding:
    def test_total_dim_matches_config(self):
        cfg = ConditioningConfig()
        registry = _registry_full()
        embed = AttributeEmbedding(cfg, registry)
        assert embed.total_dim == 16 + 16 + 8 + 12 + 12

    def test_forward_shape(self):
        cfg = ConditioningConfig()
        registry = _registry_full()
        embed = AttributeEmbedding(cfg, registry)
        labels = {
            "language": torch.tensor([0, 1]),
            "emotion": torch.tensor([0, 1]),
            "gender": torch.tensor([0, 1]),
            "age": torch.tensor([0, 1]),
            "region": torch.tensor([0, 1]),
        }
        out = embed(labels, batch_size=2, device=torch.device("cpu"))
        assert out.shape == (2, embed.total_dim)

    def test_broadcast_over_time(self):
        cfg = ConditioningConfig()
        registry = _registry_full()
        embed = AttributeEmbedding(cfg, registry)
        attr = torch.zeros(3, embed.total_dim)
        broadcast = embed.broadcast_over_time(attr, time_steps=17)
        assert broadcast.shape == (3, 17, embed.total_dim)

    def test_missing_labels_default_to_zero(self):
        cfg = ConditioningConfig()
        registry = _registry_full()
        embed = AttributeEmbedding(cfg, registry)
        out = embed({"language": torch.tensor([0, 1])}, batch_size=2, device=torch.device("cpu"))
        assert out.shape == (2, embed.total_dim)


class TestPrependPromptCEConditioner:
    def _build(self, encoder_input_dim=80, encoder_hidden_dim=512):
        cfg = ConditioningConfig(strategy=ConditioningStrategy.PREPEND_PROMPT_CE)
        mt_cfg = MultiTaskConfig()
        registry = _registry_legacy()
        cond = PrependPromptCEConditioner(cfg, mt_cfg, registry)
        cond.initialize_layers(encoder_input_dim, encoder_hidden_dim)
        return cond

    def test_apply_input_prepends_positions(self):
        cond = self._build()
        features = torch.randn(2, 50, 80)
        lengths = torch.tensor([50, 50])
        labels = {"language": torch.tensor([0, 1])}
        augmented, augmented_lens, state = cond.apply_input(features, lengths, labels)
        assert state.num_prepended == cond.num_positions
        assert augmented.shape == (2, 50 + cond.num_positions, 80)
        assert torch.equal(augmented_lens, lengths + cond.num_positions)

    def test_aux_loss_finite(self):
        cond = self._build()
        features = torch.randn(2, 20, 80)
        lengths = torch.tensor([20, 20])
        labels = {
            t: torch.zeros(2, dtype=torch.long)
            for t in MultiTaskConfig().task_order
        }
        augmented, _, state = cond.apply_input(features, lengths, labels)
        encoder_out = torch.randn(2, augmented.shape[1], 512)
        loss, metrics = cond.compute_aux_loss(encoder_out, labels, state)
        assert torch.isfinite(loss).all()
        assert all(k.startswith("acc_") for k in metrics)

    def test_strip_speech(self):
        cond = self._build()
        features = torch.randn(2, 30, 80)
        lengths = torch.tensor([30, 30])
        labels = {"language": torch.tensor([0, 1])}
        augmented, augmented_lens, state = cond.apply_input(features, lengths, labels)
        encoder_out = torch.randn(2, augmented.shape[1], 512)
        encoder_lens = augmented_lens
        speech, speech_lens = cond.strip_speech(encoder_out, encoder_lens, state)
        assert speech.shape == (2, 30, 512)
        assert torch.equal(speech_lens, torch.tensor([30, 30]))

    def test_predict_attributes(self):
        cond = self._build()
        features = torch.randn(2, 20, 80)
        lengths = torch.tensor([20, 20])
        labels = {"language": torch.tensor([0, 1])}
        augmented, _, state = cond.apply_input(features, lengths, labels)
        encoder_out = torch.randn(2, augmented.shape[1], 512)
        preds = cond.predict_attributes(encoder_out, state)
        assert preds is not None
        for task in MultiTaskConfig().task_order:
            assert task in preds
            assert preds[task].shape == (2,)


class TestFeatureConcatConditioner:
    def _build(self, encoder_input_dim=80, encoder_hidden_dim=512):
        cfg = ConditioningConfig(strategy=ConditioningStrategy.FEATURE_CONCAT)
        mt_cfg = MultiTaskConfig.full_attributes()
        registry = _registry_full()
        cond = FeatureConcatConditioner(cfg, mt_cfg, registry)
        cond.initialize_layers(encoder_input_dim, encoder_hidden_dim)
        return cond

    def test_shape_preserved(self):
        cond = self._build()
        features = torch.randn(3, 40, 80)
        lengths = torch.tensor([40, 40, 40])
        labels = {
            "language": torch.tensor([0, 1, 2]),
            "emotion": torch.tensor([0, 1, 2]),
            "gender": torch.tensor([0, 1, 0]),
            "age": torch.tensor([0, 1, 2]),
            "region": torch.tensor([0, 1, 2]),
        }
        projected, projected_lens, state = cond.apply_input(features, lengths, labels)
        assert projected.shape == (3, 40, 80)
        assert torch.equal(projected_lens, lengths)
        assert state.num_prepended == 0
        assert state.feature_concat_applied

    def test_no_prompt_decoding(self):
        cond = self._build()
        features = torch.randn(2, 20, 80)
        lengths = torch.tensor([20, 20])
        labels = {"language": torch.tensor([0, 1])}
        projected, _, state = cond.apply_input(features, lengths, labels)
        encoder_out = torch.randn(2, projected.shape[1], 512)
        assert cond.predict_attributes(encoder_out, state) is None

    def test_strip_no_op(self):
        cond = self._build()
        features = torch.randn(2, 10, 80)
        lengths = torch.tensor([10, 10])
        labels = {"language": torch.tensor([0, 1])}
        projected, projected_lens, state = cond.apply_input(features, lengths, labels)
        encoder_out = torch.randn(2, projected.shape[1], 512)
        speech, speech_lens = cond.strip_speech(encoder_out, projected_lens, state)
        assert speech.shape == encoder_out.shape


class TestDecoderTagOnlyConditioner:
    def _build(self):
        cfg = ConditioningConfig(strategy=ConditioningStrategy.DECODER_TAG_ONLY)
        mt_cfg = MultiTaskConfig.full_attributes()
        registry = _registry_full()
        cond = DecoderTagOnlyConditioner(cfg, mt_cfg, registry)
        cond.initialize_layers(80, 512)
        return cond

    def test_passthrough(self):
        cond = self._build()
        features = torch.randn(2, 30, 80)
        lengths = torch.tensor([30, 30])
        out, out_lens, state = cond.apply_input(features, lengths, None)
        assert torch.equal(out, features)
        assert torch.equal(out_lens, lengths)
        assert state.num_prepended == 0
        assert not state.feature_concat_applied

    def test_decorate_target_appends_tags(self):
        cond = self._build()
        decorated = cond.decorate_target("Xin chào.", {"language": 0, "emotion": 1})
        assert "<vi-VN>" in decorated
        assert "<emo:happy>" in decorated

    def test_parse_target_roundtrip(self):
        cond = self._build()
        decorated = cond.decorate_target("Xin chào.", {"language": 0, "emotion": 1})
        stripped, attrs = cond.parse_target_tags(decorated)
        assert stripped == "Xin chào."
        assert attrs == {"language": 0, "emotion": 1}


class TestHybridConditioner:
    def _build(self):
        cfg = ConditioningConfig(strategy=ConditioningStrategy.HYBRID)
        mt_cfg = MultiTaskConfig.full_attributes()
        registry = _registry_full()
        cond = HybridConditioner(cfg, mt_cfg, registry)
        cond.initialize_layers(80, 512)
        return cond

    def test_feature_concat_path(self):
        cond = self._build()
        features = torch.randn(2, 20, 80)
        lengths = torch.tensor([20, 20])
        labels = {a: torch.tensor([0, 1]) for a in cond.config.attribute_order}
        projected, projected_lens, state = cond.apply_input(features, lengths, labels)
        assert projected.shape == (2, 20, 80)
        assert state.feature_concat_applied

    def test_aux_loss_with_labels(self):
        cond = self._build()
        encoder_out = torch.randn(2, 20, 512)
        labels = {a: torch.tensor([0, 1]) for a in cond.config.attribute_order}
        loss, metrics = cond.compute_aux_loss(encoder_out, labels, state=cond.config)
        assert torch.isfinite(loss).all()
        assert any(k.startswith("acc_") for k in metrics)

    def test_aux_loss_disabled_returns_zero(self):
        cfg = ConditioningConfig(
            strategy=ConditioningStrategy.HYBRID, enable_aux_head_loss=False
        )
        mt_cfg = MultiTaskConfig.full_attributes()
        registry = _registry_full()
        cond = HybridConditioner(cfg, mt_cfg, registry)
        cond.initialize_layers(80, 512)
        encoder_out = torch.randn(2, 20, 512)
        labels = {a: torch.tensor([0, 1]) for a in cfg.attribute_order}
        loss, metrics = cond.compute_aux_loss(encoder_out, labels, state=cfg)
        assert loss.item() == 0.0
        assert metrics == {}

    def test_emits_decoder_tags_flag(self):
        cond = self._build()
        assert cond.emits_decoder_tags
        decorated = cond.decorate_target("Hi.", {"language": 0})
        assert "<vi-VN>" in decorated


class TestAuxLossScheduler:
    def test_linear_decay(self):
        sched = AuxLossScheduler(initial_weight=1.0, decay_epochs=4, min_weight=0.0)
        assert sched.weight(0) == 1.0
        assert sched.weight(2) == pytest.approx(0.5)
        assert sched.weight(4) == 0.0
        assert sched.weight(10) == 0.0

    def test_min_weight_floor(self):
        sched = AuxLossScheduler(initial_weight=1.0, decay_epochs=4, min_weight=0.2)
        assert sched.weight(100) == 0.2

    def test_zero_decay_epochs_returns_min(self):
        sched = AuxLossScheduler(initial_weight=1.0, decay_epochs=0, min_weight=0.0)
        assert sched.weight(0) == 0.0

    def test_invalid_config(self):
        with pytest.raises(ValueError):
            AuxLossScheduler(initial_weight=-1.0)
        with pytest.raises(ValueError):
            AuxLossScheduler(initial_weight=0.5, min_weight=1.0)

    def test_is_active(self):
        sched = AuxLossScheduler(initial_weight=1.0, decay_epochs=4)
        assert sched.is_active(0)
        assert not sched.is_active(100)


class TestGradientFlow:
    def test_feature_concat_gradient_flows(self):
        cfg = ConditioningConfig(strategy=ConditioningStrategy.FEATURE_CONCAT)
        mt_cfg = MultiTaskConfig.full_attributes()
        registry = _registry_full()
        cond = FeatureConcatConditioner(cfg, mt_cfg, registry)
        cond.initialize_layers(80, 512)

        features = torch.randn(2, 20, 80, requires_grad=True)
        lengths = torch.tensor([20, 20])
        labels = {a: torch.tensor([0, 1]) for a in cfg.attribute_order}
        projected, _, _ = cond.apply_input(features, lengths, labels)
        loss = projected.sum()
        loss.backward()
        assert features.grad is not None
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in cond.parameters()
        )

    def test_prepend_prompt_gradient_flows(self):
        cfg = ConditioningConfig(strategy=ConditioningStrategy.PREPEND_PROMPT_CE)
        mt_cfg = MultiTaskConfig()
        registry = _registry_legacy()
        cond = PrependPromptCEConditioner(cfg, mt_cfg, registry)
        cond.initialize_layers(80, 512)

        features = torch.randn(2, 20, 80, requires_grad=True)
        lengths = torch.tensor([20, 20])
        labels = {"language": torch.tensor([0, 1])}
        augmented, _, state = cond.apply_input(features, lengths, labels)
        encoder_out = torch.randn(2, augmented.shape[1], 512, requires_grad=True)
        loss, _ = cond.compute_aux_loss(
            encoder_out,
            {t: torch.zeros(2, dtype=torch.long) for t in mt_cfg.task_order},
            state,
        )
        loss.backward()
        assert encoder_out.grad is not None
