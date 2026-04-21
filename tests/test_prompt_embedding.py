import pytest
import torch

from multitalker_asr.configs.multitask import MultiTaskConfig
from multitalker_asr.models.prompt_embedding import TaskTokenRegistry, PromptEmbedding


class TestMultiTaskConfig:
    def test_default_config(self):
        cfg = MultiTaskConfig()
        assert cfg.num_prompt_positions == 6
        assert cfg.prompt_embed_dim == 80
        assert cfg.emotion_classes == 7
        assert cfg.gender_classes == 2
        assert cfg.age_classes == 4
        assert cfg.voice_state_classes == 2
        assert cfg.language_classes == 4
        assert cfg.textnorm_classes == 2

    def test_total_prompt_tokens(self):
        cfg = MultiTaskConfig()
        # 4 + 7 + 2 + 4 + 2 + 2 = 21
        assert cfg.total_prompt_tokens == 21

    def test_task_class_counts(self):
        cfg = MultiTaskConfig()
        counts = cfg.task_class_counts
        assert counts["language"] == 4
        assert counts["emotion"] == 7
        assert counts["gender"] == 2
        assert counts["age"] == 4
        assert counts["voice_state"] == 2
        assert counts["textnorm"] == 2

    def test_invalid_embed_dim(self):
        with pytest.raises(ValueError, match="prompt_embed_dim must be > 0"):
            MultiTaskConfig(prompt_embed_dim=0)

    def test_invalid_class_count(self):
        with pytest.raises(ValueError, match="Class count for 'emotion' must be > 0"):
            MultiTaskConfig(emotion_classes=0)

    def test_invalid_encoder_source(self):
        with pytest.raises(ValueError, match="encoder_source must be"):
            MultiTaskConfig(encoder_source="invalid")

    def test_task_order_mismatch(self):
        with pytest.raises(ValueError, match="task_order length"):
            MultiTaskConfig(task_order=["language", "emotion"])


class TestTaskTokenRegistry:
    def test_contiguous_ids(self):
        cfg = MultiTaskConfig()
        registry = TaskTokenRegistry(cfg)
        # language: [0,3], emotion: [4,10], gender: [11,12], age: [13,16], voice_state: [17,18], textnorm: [19,20]
        assert registry.get_embed_id("language", 0) == 0
        assert registry.get_embed_id("language", 3) == 3
        assert registry.get_embed_id("emotion", 0) == 4
        assert registry.get_embed_id("emotion", 6) == 10
        assert registry.get_embed_id("gender", 0) == 11
        assert registry.get_embed_id("gender", 1) == 12
        assert registry.get_embed_id("age", 0) == 13
        assert registry.get_embed_id("age", 3) == 16
        assert registry.get_embed_id("voice_state", 0) == 17
        assert registry.get_embed_id("voice_state", 1) == 18
        assert registry.get_embed_id("textnorm", 0) == 19
        assert registry.get_embed_id("textnorm", 1) == 20

    def test_total_tokens(self):
        cfg = MultiTaskConfig()
        registry = TaskTokenRegistry(cfg)
        assert registry.total_tokens == 21

    def test_no_id_overlap(self):
        cfg = MultiTaskConfig()
        registry = TaskTokenRegistry(cfg)
        all_ids = set()
        for task in registry.task_names():
            count = registry.get_class_count(task)
            for i in range(count):
                embed_id = registry.get_embed_id(task, i)
                assert embed_id not in all_ids, f"Duplicate ID {embed_id} for {task}:{i}"
                all_ids.add(embed_id)
        assert len(all_ids) == registry.total_tokens

    def test_task_names(self):
        cfg = MultiTaskConfig()
        registry = TaskTokenRegistry(cfg)
        assert registry.task_names() == [
            "language", "emotion", "gender", "age", "voice_state", "textnorm"
        ]

    def test_invalid_task(self):
        cfg = MultiTaskConfig()
        registry = TaskTokenRegistry(cfg)
        with pytest.raises(ValueError, match="Unknown task"):
            registry.get_embed_id("nonexistent", 0)

    def test_invalid_class_idx(self):
        cfg = MultiTaskConfig()
        registry = TaskTokenRegistry(cfg)
        with pytest.raises(ValueError, match="class_idx .* out of range"):
            registry.get_embed_id("gender", 5)

    def test_negative_class_idx(self):
        cfg = MultiTaskConfig()
        registry = TaskTokenRegistry(cfg)
        with pytest.raises(ValueError, match="class_idx .* out of range"):
            registry.get_embed_id("emotion", -1)

    def test_label_name_lookup(self):
        cfg = MultiTaskConfig()
        registry = TaskTokenRegistry(cfg)
        assert registry.get_label_name("emotion", 0) == "happy"
        assert registry.get_label_name("gender", 1) == "female"
        assert registry.get_label_name("age", 2) == "middle_age"
        assert registry.get_label_name("voice_state", 1) == "drunk"

    def test_class_idx_from_label(self):
        cfg = MultiTaskConfig()
        registry = TaskTokenRegistry(cfg)
        assert registry.get_class_idx("emotion", "angry") == 2
        assert registry.get_class_idx("language", "vi") == 0
        with pytest.raises(ValueError, match="Unknown label"):
            registry.get_class_idx("emotion", "excited")


class TestPromptEmbedding:
    def test_output_shape(self):
        cfg = MultiTaskConfig()
        prompt = PromptEmbedding(cfg)
        labels = {
            "language": torch.tensor([0, 1]),
            "emotion": torch.tensor([3, 0]),
            "gender": torch.tensor([0, 1]),
            "age": torch.tensor([1, 2]),
            "voice_state": torch.tensor([0, 0]),
            "textnorm": torch.tensor([0, 1]),
        }
        out = prompt(labels)
        assert out.shape == (2, 6, 80)

    def test_missing_task_uses_default(self):
        cfg = MultiTaskConfig()
        prompt = PromptEmbedding(cfg)
        # Only provide language — rest default to 0
        labels = {"language": torch.tensor([0, 1])}
        out = prompt(labels)
        assert out.shape == (2, 6, 80)

    def test_empty_labels_raises(self):
        cfg = MultiTaskConfig()
        prompt = PromptEmbedding(cfg)
        with pytest.raises(RuntimeError, match="Cannot infer batch size"):
            prompt({})

    def test_none_labels_raises(self):
        cfg = MultiTaskConfig()
        prompt = PromptEmbedding(cfg)
        with pytest.raises(RuntimeError, match="Cannot infer batch size"):
            prompt(None)

    def test_gradient_flows(self):
        cfg = MultiTaskConfig()
        prompt = PromptEmbedding(cfg)
        labels = {"language": torch.tensor([0, 1])}
        out = prompt(labels)
        loss = out.sum()
        loss.backward()
        assert prompt.embed.weight.grad is not None
        assert prompt.embed.weight.grad.abs().sum() > 0

    def test_different_labels_different_outputs(self):
        cfg = MultiTaskConfig()
        prompt = PromptEmbedding(cfg)
        labels_a = {"emotion": torch.tensor([0])}
        labels_b = {"emotion": torch.tensor([1])}
        out_a = prompt(labels_a)
        out_b = prompt(labels_b)
        # Emotion is position 1, should differ
        assert not torch.allclose(out_a[0, 1], out_b[0, 1])

    def test_registry_property(self):
        cfg = MultiTaskConfig()
        prompt = PromptEmbedding(cfg)
        assert prompt.registry.total_tokens == 21
        assert prompt.num_positions == 6

    def test_single_batch(self):
        cfg = MultiTaskConfig()
        prompt = PromptEmbedding(cfg)
        labels = {"language": torch.tensor([2])}
        out = prompt(labels)
        assert out.shape == (1, 6, 80)
