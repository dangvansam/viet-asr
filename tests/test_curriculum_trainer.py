import pytest
import torch
import torch.nn as nn

from multitalker_asr.training.losses.dynamic_weighting import DynamicLossWeighting
from multitalker_asr.training.curriculum_trainer import CurriculumPhase, PHASE_DEFAULTS


class TestDynamicLossWeighting:
    def test_two_tasks_weighted(self):
        weighting = DynamicLossWeighting(task_names=["rnnt", "prompt"])
        losses = {"rnnt": torch.tensor(2.0), "prompt": torch.tensor(3.0)}
        total, stats = weighting(losses)
        assert not torch.isnan(total)
        assert "loss_rnnt" in stats
        assert "loss_prompt" in stats
        assert "weight_rnnt" in stats
        assert "weight_prompt" in stats

    def test_single_task_passthrough(self):
        weighting = DynamicLossWeighting(task_names=["rnnt"])
        losses = {"rnnt": torch.tensor(5.0)}
        total, stats = weighting(losses)
        assert not torch.isnan(total)
        assert stats["loss_rnnt"] == pytest.approx(5.0)

    def test_nan_loss_skipped(self):
        weighting = DynamicLossWeighting(task_names=["rnnt", "prompt"])
        losses = {"rnnt": torch.tensor(float("nan")), "prompt": torch.tensor(1.0)}
        total, stats = weighting(losses)
        assert not torch.isnan(total)
        assert "loss_rnnt" not in stats  # NaN was skipped

    def test_log_vars_are_learnable(self):
        weighting = DynamicLossWeighting(task_names=["rnnt", "prompt"])
        losses = {"rnnt": torch.tensor(2.0), "prompt": torch.tensor(3.0)}
        total, _ = weighting(losses)
        total.backward()
        for name, param in weighting.log_vars.items():
            assert param.grad is not None

    def test_log_var_clamping(self):
        weighting = DynamicLossWeighting(task_names=["rnnt"])
        # Set log_var to extreme value
        with torch.no_grad():
            weighting.log_vars["rnnt"].fill_(100.0)
        losses = {"rnnt": torch.tensor(1.0)}
        total, stats = weighting(losses)
        # Should be clamped to [-6, 6]
        assert stats["log_var_rnnt"] == pytest.approx(6.0)

    def test_missing_task_in_losses(self):
        weighting = DynamicLossWeighting(task_names=["rnnt", "prompt"])
        losses = {"rnnt": torch.tensor(1.0)}  # "prompt" missing
        total, stats = weighting(losses)
        assert not torch.isnan(total)
        assert "loss_prompt" not in stats


class TestCurriculumPhase:
    def test_enum_values(self):
        assert CurriculumPhase.ASR.value == "asr"
        assert CurriculumPhase.MULTITALKER.value == "multitalker"
        assert CurriculumPhase.PARALINGUISTIC.value == "paralinguistic"

    def test_phase_defaults_exist(self):
        for phase in CurriculumPhase:
            assert phase in PHASE_DEFAULTS
            defaults = PHASE_DEFAULTS[phase]
            assert "num_frozen_layers" in defaults
            assert "learning_rate" in defaults
            assert "enable_prompt_loss" in defaults

    def test_asr_phase_no_prompt(self):
        defaults = PHASE_DEFAULTS[CurriculumPhase.ASR]
        assert defaults["enable_prompt_loss"] is False
        assert defaults["num_frozen_layers"] == 18

    def test_paralinguistic_phase_prompt_active(self):
        defaults = PHASE_DEFAULTS[CurriculumPhase.PARALINGUISTIC]
        assert defaults["enable_prompt_loss"] is True
        assert defaults["num_frozen_layers"] == 0


class TestFreezeEncoderLayers:
    def test_freeze_layers(self):
        """Test freezing logic with a mock encoder."""
        layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])

        # Freeze first 3
        n = 3
        for layer in layers[:n]:
            for param in layer.parameters():
                param.requires_grad = False

        frozen = sum(1 for l in layers if not any(p.requires_grad for p in l.parameters()))
        assert frozen == 3

        # Remaining should be trainable
        trainable = sum(1 for l in layers if any(p.requires_grad for p in l.parameters()))
        assert trainable == 2

    def test_freeze_more_than_total(self):
        """Freezing more layers than exist should freeze all."""
        layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])
        n = min(10, len(layers))
        for layer in layers[:n]:
            for param in layer.parameters():
                param.requires_grad = False

        frozen = sum(1 for l in layers if not any(p.requires_grad for p in l.parameters()))
        assert frozen == 3
