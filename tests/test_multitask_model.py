import pytest
import torch
import torch.nn as nn

from multitalker_asr.configs.multitask import MultiTaskConfig
from multitalker_asr.models.prompt_embedding import PromptEmbedding, TaskTokenRegistry
from multitalker_asr.training.losses.multi_task_loss import MultiTaskLoss


class TestMultiTaskLoss:
    def test_weighted_combination(self):
        loss = MultiTaskLoss(rnnt_weight=1.0, prompt_weight=0.5)
        l_rnnt = torch.tensor(2.0)
        l_prompt = torch.tensor(4.0)
        total, stats = loss(l_rnnt, l_prompt)
        assert torch.isclose(total, torch.tensor(4.0))  # 1.0*2 + 0.5*4
        assert stats["loss_rnnt"] == pytest.approx(2.0)
        assert stats["loss_prompt"] == pytest.approx(4.0)

    def test_nan_rnnt_replaced(self):
        loss = MultiTaskLoss()
        total, stats = loss(torch.tensor(float("nan")), torch.tensor(1.0))
        assert not torch.isnan(total)
        assert stats["loss_rnnt"] == pytest.approx(0.0)

    def test_nan_prompt_replaced(self):
        loss = MultiTaskLoss()
        total, stats = loss(torch.tensor(1.0), torch.tensor(float("nan")))
        assert not torch.isnan(total)
        assert stats["loss_prompt"] == pytest.approx(0.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="weights must be >= 0"):
            MultiTaskLoss(rnnt_weight=-1.0)

    def test_zero_weights(self):
        loss = MultiTaskLoss(rnnt_weight=0.0, prompt_weight=0.0)
        total, _ = loss(torch.tensor(5.0), torch.tensor(3.0))
        assert total.item() == pytest.approx(0.0)


class TestPrependPrompts:
    """Test _prepend_prompts logic using PromptEmbedding directly (no NeMo dependency)."""

    def test_shape(self):
        cfg = MultiTaskConfig()
        prompt = PromptEmbedding(cfg)
        speech = torch.randn(2, 100, 80)
        labels = {"language": torch.tensor([0, 1])}
        embeds = prompt(labels)  # [2, 6, 80]
        augmented = torch.cat([embeds, speech], dim=1)
        assert augmented.shape == (2, 106, 80)

    def test_num_prepended(self):
        cfg = MultiTaskConfig()
        prompt = PromptEmbedding(cfg)
        assert prompt.num_positions == 6


class TestComputePromptLoss:
    """Test prompt loss computation with mock encoder output."""

    def test_loss_shape_and_accuracy(self):
        cfg = MultiTaskConfig()
        registry = TaskTokenRegistry(cfg)
        classifier = nn.Linear(64, registry.total_tokens)
        loss_fn = nn.CrossEntropyLoss()

        # Simulate encoder output for prompt positions
        encoder_out = torch.randn(2, 6, 64)
        logits = classifier(encoder_out)  # [2, 6, 21]

        labels = {
            "language": torch.tensor([0, 1]),
            "emotion": torch.tensor([3, 0]),
            "gender": torch.tensor([0, 1]),
            "age": torch.tensor([1, 2]),
            "voice_state": torch.tensor([0, 0]),
            "textnorm": torch.tensor([0, 1]),
        }

        total_loss = torch.tensor(0.0)
        accuracy_dict = {}

        for i, task in enumerate(cfg.task_order):
            task_logits = logits[:, i, :]  # [2, 21]
            targets = labels[task]
            start, _ = registry.get_task_range(task)
            targets_shifted = targets + start
            loss_i = loss_fn(task_logits, targets_shifted)
            total_loss = total_loss + loss_i

            with torch.no_grad():
                preds = task_logits.argmax(dim=-1)
                acc = (preds == targets_shifted).float().mean().item()
                accuracy_dict[f"acc_{task}"] = acc

        total_loss = total_loss / 6
        assert total_loss.shape == ()
        assert len(accuracy_dict) == 6
        assert all(0.0 <= v <= 1.0 for v in accuracy_dict.values())

    def test_gradient_flows_through_classifier(self):
        cfg = MultiTaskConfig()
        registry = TaskTokenRegistry(cfg)
        classifier = nn.Linear(64, registry.total_tokens)
        loss_fn = nn.CrossEntropyLoss()

        encoder_out = torch.randn(2, 6, 64, requires_grad=True)
        logits = classifier(encoder_out)

        labels = {"language": torch.tensor([0, 1])}
        start, _ = registry.get_task_range("language")
        targets = labels["language"] + start

        loss = loss_fn(logits[:, 0, :], targets)
        loss.backward()
        assert encoder_out.grad is not None


class TestPredictPromptLabels:
    """Test prompt label prediction (inference mode)."""

    def test_predict_returns_all_tasks(self):
        cfg = MultiTaskConfig()
        registry = TaskTokenRegistry(cfg)
        classifier = nn.Linear(64, registry.total_tokens)

        encoder_out = torch.randn(2, 6, 64)
        logits = classifier(encoder_out)

        predictions = {}
        for i, task in enumerate(cfg.task_order):
            task_logits = logits[:, i, :]
            start, end = registry.get_task_range(task)
            task_specific = task_logits[:, start:end]
            predictions[task] = task_specific.argmax(dim=-1)

        assert set(predictions.keys()) == set(cfg.task_order)
        for task in cfg.task_order:
            pred = predictions[task]
            assert pred.shape == (2,)
            max_class = registry.get_class_count(task)
            assert (pred >= 0).all() and (pred < max_class).all()


class TestPromptClassifierDim:
    def test_classifier_matches_total_tokens(self):
        cfg = MultiTaskConfig()
        registry = TaskTokenRegistry(cfg)
        encoder_hidden = 256
        classifier = nn.Linear(encoder_hidden, registry.total_tokens)
        assert classifier.out_features == 21
        assert classifier.in_features == 256
