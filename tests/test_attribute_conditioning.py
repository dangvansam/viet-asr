import json
import tempfile
from pathlib import Path

import pytest

from multitalker_asr.configs import (
    DEFAULT_ATTRIBUTE_VOCABULARY,
    AttributeAxis,
    AttributeVocabulary,
    ConditionerFactory,
    ConditioningConfig,
    ConditioningStrategy,
    MultiTaskConfig,
    TagEmissionMode,
)
from multitalker_asr.data import (
    SCHEMA_VERSION,
    AttributeConfidence,
    ManifestMigrator,
    ManifestRecord,
    TypedManifestReader,
)
from multitalker_asr.models.prompt_embedding import TaskTokenRegistry


class TestAttributeVocabulary:
    def test_default_axes(self):
        vocab = DEFAULT_ATTRIBUTE_VOCABULARY
        assert [a.name for a in vocab.axes()] == [
            "language",
            "emotion",
            "gender",
            "age",
            "region",
        ]

    def test_class_counts(self):
        vocab = DEFAULT_ATTRIBUTE_VOCABULARY
        assert vocab.class_counts() == {
            "language": 4,
            "emotion": 7,
            "gender": 2,
            "age": 4,
            "region": 3,
        }

    def test_total_classes(self):
        assert DEFAULT_ATTRIBUTE_VOCABULARY.total_classes() == 20

    def test_tag_format(self):
        vocab = DEFAULT_ATTRIBUTE_VOCABULARY
        assert vocab.language.tag("vi-VN") == "<vi-VN>"
        assert vocab.emotion.tag("happy") == "<emo:happy>"
        assert vocab.gender.tag("male") == "<gen:male>"
        assert vocab.region.tag("northern") == "<reg:northern>"

    def test_all_tag_strings_unique(self):
        tags = DEFAULT_ATTRIBUTE_VOCABULARY.all_tag_strings()
        assert len(tags) == 20
        assert len(set(tags)) == 20

    def test_tag_to_axis_label_roundtrip(self):
        vocab = DEFAULT_ATTRIBUTE_VOCABULARY
        for axis in vocab.axes():
            for label in axis.labels:
                tag = axis.tag(label)
                result = vocab.tag_to_axis_label(tag)
                assert result == (axis.name, label)

    def test_tag_to_axis_label_unknown(self):
        assert DEFAULT_ATTRIBUTE_VOCABULARY.tag_to_axis_label("<nonexistent>") is None
        assert DEFAULT_ATTRIBUTE_VOCABULARY.tag_to_axis_label("not_a_tag") is None

    def test_axis_index_of_unknown(self):
        with pytest.raises(ValueError, match="Unknown label"):
            DEFAULT_ATTRIBUTE_VOCABULARY.language.index_of("kr-KR")


class TestConditioningConfig:
    def test_default_strategy(self):
        cfg = ConditioningConfig()
        assert cfg.strategy == ConditioningStrategy.HYBRID
        assert cfg.tag_emission_mode == TagEmissionMode.AUTO

    def test_string_strategy_coerced(self):
        cfg = ConditioningConfig(strategy="feature_concat")
        assert cfg.strategy == ConditioningStrategy.FEATURE_CONCAT

    def test_total_attribute_dim(self):
        cfg = ConditioningConfig()
        assert cfg.total_attribute_dim == 16 + 16 + 8 + 12 + 12

    def test_strategy_flags(self):
        hybrid = ConditioningConfig(strategy="hybrid")
        assert hybrid.uses_feature_concat
        assert hybrid.emits_decoder_tags
        assert not hybrid.uses_prepend_prompt

        prepend = ConditioningConfig(strategy="prepend_prompt_ce")
        assert prepend.uses_prepend_prompt
        assert not prepend.uses_feature_concat
        assert not prepend.emits_decoder_tags

        tag_only = ConditioningConfig(strategy="decoder_tag_only")
        assert tag_only.emits_decoder_tags
        assert not tag_only.uses_feature_concat

        feat = ConditioningConfig(strategy="feature_concat")
        assert feat.uses_feature_concat
        assert not feat.emits_decoder_tags

    def test_invalid_p_condition(self):
        with pytest.raises(ValueError, match="p_condition"):
            ConditioningConfig(p_condition=1.5)

    def test_missing_attribute_dim(self):
        with pytest.raises(ValueError, match="missing from attribute_dims"):
            ConditioningConfig(attribute_order=["language", "unknown_axis"])

    def test_zero_attribute_dim(self):
        with pytest.raises(ValueError, match="must be > 0"):
            ConditioningConfig(attribute_dims={"language": 0, **{
                a: 8 for a in ("emotion", "gender", "age", "region")
            }})


class TestConditionerFactory:
    def test_unregistered_strategy_raises(self):
        original = ConditionerFactory._builders.pop(
            ConditioningStrategy.FEATURE_CONCAT, None
        )
        try:
            cfg = ConditioningConfig(strategy="feature_concat")
            with pytest.raises(NotImplementedError, match="No conditioner registered"):
                ConditionerFactory.build(cfg)
        finally:
            if original is not None:
                ConditionerFactory._builders[ConditioningStrategy.FEATURE_CONCAT] = original

    def test_register_and_build(self):
        sentinel = {"called": False}
        original = ConditionerFactory._builders.get(ConditioningStrategy.DECODER_TAG_ONLY)

        @ConditionerFactory.register(ConditioningStrategy.DECODER_TAG_ONLY)
        def _builder(cfg, **kwargs):
            sentinel["called"] = True
            return ("fake_conditioner", cfg)

        try:
            cfg = ConditioningConfig(strategy="decoder_tag_only")
            result = ConditionerFactory.build(cfg)
            assert sentinel["called"]
            assert result[0] == "fake_conditioner"
        finally:
            if original is not None:
                ConditionerFactory._builders[ConditioningStrategy.DECODER_TAG_ONLY] = original
            else:
                ConditionerFactory.unregister(ConditioningStrategy.DECODER_TAG_ONLY)


class TestTaskTokenRegistryFullAttributes:
    def test_full_attributes_layout(self):
        cfg = MultiTaskConfig.full_attributes()
        registry = TaskTokenRegistry(cfg, prefer_legacy_labels=False)
        assert registry.total_tokens == 4 + 7 + 2 + 4 + 3
        assert registry.task_names() == [
            "language",
            "emotion",
            "gender",
            "age",
            "region",
        ]

    def test_locale_tags_emitted(self):
        cfg = MultiTaskConfig.full_attributes()
        registry = TaskTokenRegistry(cfg, prefer_legacy_labels=False)
        assert registry.tag("language", 0) == "<vi-VN>"
        assert registry.tag("language", 1) == "<en-US>"
        assert registry.tag("emotion", 0) == "<emo:neutral>"
        assert registry.tag("region", 2) == "<reg:southern>"

    def test_parse_tag(self):
        cfg = MultiTaskConfig.full_attributes()
        registry = TaskTokenRegistry(cfg, prefer_legacy_labels=False)
        assert registry.parse_tag("<vi-VN>") == ("language", 0)
        assert registry.parse_tag("<emo:happy>") == ("emotion", 1)
        assert registry.parse_tag("<reg:southern>") == ("region", 2)
        assert registry.parse_tag("<nope>") is None

    def test_parse_tags_from_text(self):
        cfg = MultiTaskConfig.full_attributes()
        registry = TaskTokenRegistry(cfg, prefer_legacy_labels=False)
        text = "Xin chào, tôi tên là Nam.<vi-VN><emo:happy><gen:male><age:adult><reg:southern>"
        stripped, attrs = registry.parse_tags_from_text(text)
        assert stripped == "Xin chào, tôi tên là Nam."
        assert attrs == {
            "language": 0,
            "emotion": 1,
            "gender": 0,
            "age": 2,
            "region": 2,
        }

    def test_append_tags_roundtrip(self):
        cfg = MultiTaskConfig.full_attributes()
        registry = TaskTokenRegistry(cfg, prefer_legacy_labels=False)
        attrs = {"language": 0, "emotion": 1, "gender": 0, "age": 2, "region": 2}
        text = "Hello."
        out = registry.append_tags(text, attrs)
        _, recovered = registry.parse_tags_from_text(out)
        assert recovered == attrs

    def test_all_tag_strings_unique(self):
        cfg = MultiTaskConfig.full_attributes()
        registry = TaskTokenRegistry(cfg, prefer_legacy_labels=False)
        tags = registry.all_tag_strings()
        assert len(tags) == 20
        assert len(set(tags)) == 20

    def test_legacy_preserved_when_flag_true(self):
        cfg = MultiTaskConfig()
        registry = TaskTokenRegistry(cfg, prefer_legacy_labels=True)
        assert registry.get_label_name("emotion", 0) == "happy"
        assert registry.get_label_name("language", 0) == "vi"


class TestMultiTaskConfigRegion:
    def test_full_attributes_factory(self):
        cfg = MultiTaskConfig.full_attributes()
        assert cfg.num_prompt_positions == 5
        assert cfg.region_classes == 3
        assert "region" in cfg.task_order
        assert cfg.total_prompt_tokens == 4 + 7 + 2 + 4 + 3

    def test_region_class_count(self):
        cfg = MultiTaskConfig.full_attributes()
        assert cfg._class_count_for("region") == 3


class TestManifestRecord:
    def test_legacy_forward_compat(self):
        legacy = {
            "audio_filepath": "/x.wav",
            "offset": 0.0,
            "duration": 2.5,
            "label": "spk_0",
            "text": "hello",
            "num_speakers": 1,
        }
        rec = ManifestRecord.from_dict(legacy)
        assert rec.language is None
        assert rec.region is None
        assert rec.schema_version == 1
        assert rec.attribute_confidence.language == 0.0

    def test_full_record(self):
        full = {
            "audio_filepath": "/x.wav",
            "duration": 3.5,
            "text": "xin chào",
            "language": "vi-VN",
            "emotion": "happy",
            "gender": "female",
            "age": "adult",
            "region": "northern",
            "attribute_confidence": {"language": 0.99, "emotion": 0.7},
            "source": "vivoice",
            "schema_version": 2,
        }
        rec = ManifestRecord.from_dict(full)
        assert rec.region == "northern"
        assert rec.attribute_confidence.emotion == 0.7
        assert rec.source == "vivoice"

    def test_extra_fields_preserved(self):
        data = {"audio_filepath": "/x.wav", "duration": 1.0, "unknown_metric": 42.0}
        rec = ManifestRecord.from_dict(data)
        assert rec.extra == {"unknown_metric": 42.0}

    def test_missing_required(self):
        with pytest.raises(ValueError, match="missing required key"):
            ManifestRecord.from_dict({"audio_filepath": "/x.wav"})

    def test_to_dict_omits_empty(self):
        rec = ManifestRecord(audio_filepath="/x.wav", duration=1.0)
        d = rec.to_dict()
        assert "language" not in d
        assert "attribute_confidence" not in d
        assert "source" not in d

    def test_round_trip(self):
        full = {
            "audio_filepath": "/x.wav",
            "duration": 3.5,
            "text": "hi",
            "language": "vi-VN",
            "emotion": "happy",
            "attribute_confidence": {"language": 0.9, "emotion": 0.8},
            "source": "test",
        }
        rec = ManifestRecord.from_dict(full)
        rec2 = ManifestRecord.from_dict(rec.to_dict())
        assert rec2.language == rec.language
        assert rec2.attribute_confidence.emotion == 0.8


class TestManifestMigrator:
    def test_upgrade_file(self, tmp_path: Path):
        legacy = {
            "audio_filepath": "/a.wav",
            "offset": 0.0,
            "duration": 2.5,
            "label": "spk_0",
            "text": "old",
            "num_speakers": 1,
        }
        in_path = tmp_path / "in.json"
        out_path = tmp_path / "out.json"
        with open(in_path, "w") as f:
            f.write(json.dumps(legacy) + "\n")

        mig = ManifestMigrator(default_source="legacy")
        n = mig.upgrade_file(in_path, out_path)
        assert n == 1

        records = list(TypedManifestReader(out_path))
        assert len(records) == 1
        assert records[0].schema_version == SCHEMA_VERSION
        assert records[0].source == "legacy"
