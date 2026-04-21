import json
import os
import tempfile

import pytest
import torch

from multitalker_asr.configs.multitask import MultiTaskConfig
from multitalker_asr.models.prompt_embedding import TaskTokenRegistry
from multitalker_asr.data.datasets.multitask import MultitaskStreamingDataset, DEFAULT_TASK_LABELS
from multitalker_asr.data.collators.multitask import MultitaskCollator


@pytest.fixture
def task_registry():
    return TaskTokenRegistry(MultiTaskConfig())


@pytest.fixture
def manifest_with_labels(tmp_path):
    """Create a manifest JSONL with all task fields."""
    manifest = tmp_path / "manifest.json"
    entries = [
        {
            "audio_filepath": "/tmp/dummy1.wav",
            "text": "xin chao",
            "duration": 2.0,
            "emotion": "happy",
            "gender": "female",
            "age": "young",
            "voice_state": "sober",
            "language": "vi",
        },
        {
            "audio_filepath": "/tmp/dummy2.wav",
            "text": "tam biet",
            "duration": 1.5,
            "emotion": "sad",
            "gender": "male",
            "age": "old",
            "voice_state": "drunk",
            "language": "vi",
        },
        {
            "audio_filepath": "/tmp/dummy3.wav",
            "text": "hello world",
            "duration": 1.0,
            "emotion": "neutral",
            "gender": "male",
            "age": "child",
            "voice_state": "sober",
            "language": "en",
        },
    ]
    with open(manifest, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return str(manifest)


@pytest.fixture
def manifest_without_labels(tmp_path):
    """Create a manifest JSONL without task fields (old format)."""
    manifest = tmp_path / "manifest_old.json"
    entries = [
        {"audio_filepath": "/tmp/dummy1.wav", "text": "xin chao", "duration": 2.0},
        {"audio_filepath": "/tmp/dummy2.wav", "text": "tam biet", "duration": 1.5},
    ]
    with open(manifest, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return str(manifest)


class TestParseTaskLabels:
    def test_full_labels(self, task_registry):
        entry = {
            "emotion": "happy",
            "gender": "female",
            "age": "young",
            "voice_state": "sober",
            "language": "vi",
            "textnorm": "with_itn",
        }
        dataset = MultitaskStreamingDataset.__new__(MultitaskStreamingDataset)
        dataset._task_registry = task_registry
        dataset._default_labels = DEFAULT_TASK_LABELS

        labels = dataset._parse_task_labels(entry)
        assert labels["emotion"] == 0  # happy = 0
        assert labels["gender"] == 1  # female = 1
        assert labels["age"] == 1  # young = 1
        assert labels["voice_state"] == 0  # sober = 0
        assert labels["language"] == 0  # vi = 0

    def test_missing_fields_use_defaults(self, task_registry):
        entry = {"emotion": "angry"}  # Only emotion provided
        dataset = MultitaskStreamingDataset.__new__(MultitaskStreamingDataset)
        dataset._task_registry = task_registry
        dataset._default_labels = DEFAULT_TASK_LABELS

        labels = dataset._parse_task_labels(entry)
        assert labels["emotion"] == 2  # angry = 2
        assert labels["gender"] == 0  # default: male = 0
        assert labels["language"] == 0  # default: vi = 0

    def test_unknown_label_uses_default(self, task_registry):
        entry = {"emotion": "excited"}  # Not in label list
        dataset = MultitaskStreamingDataset.__new__(MultitaskStreamingDataset)
        dataset._task_registry = task_registry
        dataset._default_labels = DEFAULT_TASK_LABELS

        labels = dataset._parse_task_labels(entry)
        # "excited" is unknown → falls back to default "neutral" = 3
        assert labels["emotion"] == 3

    def test_no_registry_returns_empty(self):
        entry = {"emotion": "happy"}
        dataset = MultitaskStreamingDataset.__new__(MultitaskStreamingDataset)
        dataset._task_registry = None
        dataset._default_labels = DEFAULT_TASK_LABELS

        labels = dataset._parse_task_labels(entry)
        assert labels == {}


class TestMultitaskCollator:
    def test_collates_task_labels(self):
        batch = [
            {
                "audio": __import__("numpy").zeros(16000, dtype="float32"),
                "audio_len": 16000,
                "text": "hello",
                "text_ids": [1, 2, 3],
                "duration": 1.0,
                "num_speakers": 1,
                "spk_mask": __import__("numpy").ones(16000, dtype="float32"),
                "bg_mask": __import__("numpy").zeros(16000, dtype="float32"),
                "task_labels": {"emotion": 0, "gender": 1, "age": 2},
            },
            {
                "audio": __import__("numpy").zeros(16000, dtype="float32"),
                "audio_len": 16000,
                "text": "world",
                "text_ids": [4, 5],
                "duration": 1.0,
                "num_speakers": 1,
                "spk_mask": __import__("numpy").ones(16000, dtype="float32"),
                "bg_mask": __import__("numpy").zeros(16000, dtype="float32"),
                "task_labels": {"emotion": 3, "gender": 0, "age": 1},
            },
        ]

        collator = MultitaskCollator()
        result = collator(batch)

        # Parent returns 6 elements, MultitaskCollator adds task_labels as 7th
        assert len(result) == 7
        task_labels = result[6]
        assert isinstance(task_labels, dict)
        assert torch.equal(task_labels["emotion"], torch.tensor([0, 3]))
        assert torch.equal(task_labels["gender"], torch.tensor([1, 0]))
        assert torch.equal(task_labels["age"], torch.tensor([2, 1]))

    def test_missing_task_labels_key(self):
        """Batch items without task_labels → empty dict."""
        batch = [
            {
                "audio": __import__("numpy").zeros(16000, dtype="float32"),
                "audio_len": 16000,
                "text": "hello",
                "text_ids": [1, 2],
                "duration": 1.0,
                "num_speakers": 1,
                "spk_mask": __import__("numpy").ones(16000, dtype="float32"),
                "bg_mask": __import__("numpy").zeros(16000, dtype="float32"),
            },
        ]

        collator = MultitaskCollator()
        result = collator(batch)
        task_labels = result[6]
        assert task_labels == {}


class TestMultitaskDatasetInit:
    def test_loads_manifest(self, manifest_with_labels, task_registry):
        dataset = MultitaskStreamingDataset(
            manifest_paths=[manifest_with_labels],
            task_registry=task_registry,
            max_speakers=2,
            max_samples=2,
        )
        assert len(dataset._utterances) == 3

    def test_old_manifest_works(self, manifest_without_labels, task_registry):
        """Old manifests without task fields should load without error."""
        dataset = MultitaskStreamingDataset(
            manifest_paths=[manifest_without_labels],
            task_registry=task_registry,
            max_speakers=2,
            max_samples=2,
        )
        assert len(dataset._utterances) == 2

    def test_enrich_with_task_labels(self, task_registry):
        dataset = MultitaskStreamingDataset.__new__(MultitaskStreamingDataset)
        dataset._task_registry = task_registry
        dataset._default_labels = DEFAULT_TASK_LABELS

        sample = {"audio": None, "text": "test"}
        source_utts = [{"emotion": "happy", "gender": "male"}]

        enriched = dataset._enrich_with_task_labels(sample, source_utts)
        assert "task_labels" in enriched
        assert enriched["task_labels"]["emotion"] == 0  # happy
        assert enriched["task_labels"]["gender"] == 0  # male
