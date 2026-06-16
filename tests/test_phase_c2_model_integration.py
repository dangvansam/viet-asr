from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from multitalker_asr.configs import (
    ConditioningConfig,
    ConditioningStrategy,
    ModelConfig,
    MultiTaskConfig,
)
from multitalker_asr.models.conditioning import (
    DecoderTagOnlyConditioner,
    FeatureConcatConditioner,
    HybridConditioner,
    PrependPromptCEConditioner,
)
from multitalker_asr.models.multitask_model import MultitalkerMultiTaskModel


def _make_model_without_load(
    multitask_cfg=None,
    conditioning_cfg=None,
) -> MultitalkerMultiTaskModel:
    model = MultitalkerMultiTaskModel.__new__(MultitalkerMultiTaskModel)
    model._model_cfg = ModelConfig()
    model._multitask_cfg = multitask_cfg or MultiTaskConfig()
    model._conditioning_cfg = conditioning_cfg or ConditioningConfig(
        strategy=ConditioningStrategy.PREPEND_PROMPT_CE
    )
    model._base_model = MagicMock()
    from multitalker_asr.configs import ConditionerFactory
    from multitalker_asr.models.conditioning.aux_loss_scheduler import AuxLossScheduler
    from multitalker_asr.models.prompt_embedding import TaskTokenRegistry
    from multitalker_asr.training.losses import MultiTaskLoss

    prefer_legacy = (
        model._conditioning_cfg.strategy == ConditioningStrategy.PREPEND_PROMPT_CE
    )
    model._registry = TaskTokenRegistry(
        model._multitask_cfg, prefer_legacy_labels=prefer_legacy
    )
    model._conditioner = ConditionerFactory.build(
        model._conditioning_cfg,
        multitask_config=model._multitask_cfg,
        registry=model._registry,
    )
    model._aux_scheduler = AuxLossScheduler(
        initial_weight=model._conditioning_cfg.aux_head_loss_weight,
        decay_epochs=model._conditioning_cfg.aux_head_loss_decay_epochs,
        min_weight=0.0,
    )
    model._current_epoch = 0
    model._multi_task_loss = MultiTaskLoss(
        rnnt_weight=1.0,
        prompt_weight=model._multitask_cfg.ce_loss_weight,
    )
    return model


class TestStrategyDispatch:
    def test_default_strategy_is_prepend(self):
        model = _make_model_without_load()
        assert isinstance(model.conditioner, PrependPromptCEConditioner)

    def test_decoder_tag_only_strategy(self):
        cfg = ConditioningConfig(strategy=ConditioningStrategy.DECODER_TAG_ONLY)
        model = _make_model_without_load(
            multitask_cfg=MultiTaskConfig.full_attributes(),
            conditioning_cfg=cfg,
        )
        assert isinstance(model.conditioner, DecoderTagOnlyConditioner)

    def test_feature_concat_strategy(self):
        cfg = ConditioningConfig(strategy=ConditioningStrategy.FEATURE_CONCAT)
        model = _make_model_without_load(
            multitask_cfg=MultiTaskConfig.full_attributes(),
            conditioning_cfg=cfg,
        )
        assert isinstance(model.conditioner, FeatureConcatConditioner)

    def test_hybrid_strategy(self):
        cfg = ConditioningConfig(strategy=ConditioningStrategy.HYBRID)
        model = _make_model_without_load(
            multitask_cfg=MultiTaskConfig.full_attributes(),
            conditioning_cfg=cfg,
        )
        assert isinstance(model.conditioner, HybridConditioner)


class TestPromptEmbeddingProperty:
    def test_prepend_strategy_exposes_prompt_embedding(self):
        model = _make_model_without_load()
        model._conditioner.initialize_layers(80, 512)
        assert model.prompt_embedding is not None

    def test_non_prepend_strategy_raises(self):
        cfg = ConditioningConfig(strategy=ConditioningStrategy.DECODER_TAG_ONLY)
        model = _make_model_without_load(
            multitask_cfg=MultiTaskConfig.full_attributes(),
            conditioning_cfg=cfg,
        )
        model._conditioner.initialize_layers(80, 512)
        with pytest.raises(AttributeError, match="PREPEND_PROMPT_CE"):
            _ = model.prompt_embedding


class TestEpochScheduler:
    def test_aux_loss_weight_decays(self):
        cfg = ConditioningConfig(
            strategy=ConditioningStrategy.HYBRID,
            aux_head_loss_weight=1.0,
            aux_head_loss_decay_epochs=4,
        )
        model = _make_model_without_load(
            multitask_cfg=MultiTaskConfig.full_attributes(),
            conditioning_cfg=cfg,
        )
        model.set_epoch(0)
        assert model.aux_loss_weight() == 1.0
        model.set_epoch(2)
        assert model.aux_loss_weight() == pytest.approx(0.5)
        model.set_epoch(10)
        assert model.aux_loss_weight() == 0.0

    def test_set_epoch_clamps_negative(self):
        model = _make_model_without_load()
        model.set_epoch(-5)
        assert model._current_epoch == 0


class TestPredictPromptLabels:
    def test_predict_prompt_labels_prepend(self):
        model = _make_model_without_load()
        model._conditioner.initialize_layers(80, 512)
        num_pos = model._conditioner.num_positions
        encoder_out = torch.randn(2, num_pos + 30, 512)
        preds = model.predict_prompt_labels(encoder_out)
        assert preds is not None
        for task in model._multitask_cfg.task_order:
            assert task in preds

    def test_predict_prompt_labels_feature_concat_returns_none(self):
        cfg = ConditioningConfig(strategy=ConditioningStrategy.FEATURE_CONCAT)
        model = _make_model_without_load(
            multitask_cfg=MultiTaskConfig.full_attributes(),
            conditioning_cfg=cfg,
        )
        model._conditioner.initialize_layers(80, 512)
        encoder_out = torch.randn(2, 20, 512)
        assert model.predict_prompt_labels(encoder_out) is None


class TestForwardMultitaskSimulation:
    """Simulate forward_multitask by patching the inner ASR model."""

    def _patch_asr(self, model, encoder_hidden=512, encoder_input=80):
        fake_asr = MagicMock()
        fake_asr.training = False
        fake_asr.spec_augmentation = None

        def preprocessor(input_signal, length):
            batch = input_signal.shape[0]
            return torch.randn(batch, 50, encoder_input), length

        fake_asr.preprocessor = preprocessor

        def encoder(audio_signal, length):
            return torch.randn(audio_signal.shape[0], audio_signal.shape[1], encoder_hidden), length

        fake_asr.encoder = encoder

        def decoder(targets, target_length):
            return torch.randn(targets.shape[0], targets.shape[1], encoder_hidden), target_length, None

        fake_asr.decoder = decoder

        def joint(encoder_outputs, decoder_outputs):
            B, T, _ = encoder_outputs.shape
            _, U, _ = decoder_outputs.shape
            return torch.randn(B, T, U, 100)

        fake_asr.joint = joint
        fake_asr.loss = lambda log_probs, targets, input_lengths, target_lengths: torch.tensor(
            1.5, requires_grad=True
        )

        model._base_model.asr_model = fake_asr
        return fake_asr

    def test_prepend_forward_returns_expected_keys(self):
        model = _make_model_without_load()
        model._conditioner.initialize_layers(80, 512)
        self._patch_asr(model)

        audio = torch.randn(2, 16000)
        audio_lengths = torch.tensor([16000, 16000])
        labels = {"language": torch.tensor([0, 1])}
        text = torch.zeros(2, 10, dtype=torch.long)
        text_lengths = torch.tensor([10, 10])

        out = model.forward_multitask(audio, audio_lengths, labels, text, text_lengths)
        assert "loss_total" in out
        assert "loss_rnnt" in out
        assert "loss_prompt" in out
        assert "accuracy" in out
        assert "condition_state" in out
        assert out["condition_state"].num_prepended == model._conditioner.num_positions

    def test_feature_concat_forward_no_prepend(self):
        cfg = ConditioningConfig(strategy=ConditioningStrategy.FEATURE_CONCAT)
        model = _make_model_without_load(
            multitask_cfg=MultiTaskConfig.full_attributes(),
            conditioning_cfg=cfg,
        )
        model._conditioner.initialize_layers(80, 512)
        self._patch_asr(model)

        audio = torch.randn(2, 16000)
        audio_lengths = torch.tensor([16000, 16000])
        labels = {a: torch.tensor([0, 1]) for a in cfg.attribute_order}
        text = torch.zeros(2, 10, dtype=torch.long)
        text_lengths = torch.tensor([10, 10])

        out = model.forward_multitask(audio, audio_lengths, labels, text, text_lengths)
        assert out["condition_state"].num_prepended == 0
        assert out["condition_state"].feature_concat_applied

    def test_decoder_tag_only_no_aux_loss(self):
        cfg = ConditioningConfig(strategy=ConditioningStrategy.DECODER_TAG_ONLY)
        model = _make_model_without_load(
            multitask_cfg=MultiTaskConfig.full_attributes(),
            conditioning_cfg=cfg,
        )
        model._conditioner.initialize_layers(80, 512)
        self._patch_asr(model)

        audio = torch.randn(2, 16000)
        audio_lengths = torch.tensor([16000, 16000])
        labels = {"language": torch.tensor([0, 1])}
        text = torch.zeros(2, 10, dtype=torch.long)
        text_lengths = torch.tensor([10, 10])

        out = model.forward_multitask(audio, audio_lengths, labels, text, text_lengths)
        assert out["loss_prompt"].item() == 0.0

    def test_inference_mode_no_labels(self):
        model = _make_model_without_load()
        model._conditioner.initialize_layers(80, 512)
        self._patch_asr(model)

        audio = torch.randn(1, 16000)
        audio_lengths = torch.tensor([16000])
        out = model.forward_multitask(audio, audio_lengths)
        assert "prompt_preds" in out
        assert out["loss_prompt"].item() == 0.0
        assert out["loss_rnnt"].item() == 0.0


class TestRegistryProperty:
    def test_registry_full_attributes(self):
        cfg = ConditioningConfig(strategy=ConditioningStrategy.FEATURE_CONCAT)
        model = _make_model_without_load(
            multitask_cfg=MultiTaskConfig.full_attributes(),
            conditioning_cfg=cfg,
        )
        assert model.registry.task_names() == [
            "language",
            "emotion",
            "gender",
            "age",
            "region",
        ]

    def test_registry_legacy_default(self):
        model = _make_model_without_load()
        assert model.registry.task_names() == [
            "language",
            "emotion",
            "gender",
            "age",
            "voice_state",
            "textnorm",
        ]
