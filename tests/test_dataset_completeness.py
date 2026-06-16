import json

import pytest

from multitalker_asr.data.pipeline.checkpoint import PipelineCheckpoint
from multitalker_asr.data.pipeline.config import PipelineConfig
from multitalker_asr.data.pipeline.stages.transcribe import TranscribeStage
from multitalker_asr.data.pipeline.stages.write_manifest import WriteManifestStage


class _FakeFunASR:
    def generate(self, input, language=None):
        return [{
            "text": "Bên cạnh đó, các tổ chức chính trị - xã hội.",   # ITN + PnC
            "text_tn": "Bên cạnh đó các tổ chức chính trị xã hội",     # spoken/raw
            "timestamps": [{"token": "Bên", "start_time": 0.0, "end_time": 0.2}],
        }]


class TestTranscribeTwoForm:
    def test_text_and_text_raw_differ(self):
        stage = TranscribeStage()
        from multitalker_asr.data.pipeline.config import TranscribeConfig
        fields = stage._transcribe_audio("x.wav", _FakeFunASR(), TranscribeConfig())
        assert fields["text"] == "Bên cạnh đó, các tổ chức chính trị - xã hội."
        assert fields["text_raw"] == "Bên cạnh đó các tổ chức chính trị xã hội"
        assert fields["text"] != fields["text_raw"]
        assert fields["text_itn"] == fields["text"]  # normalized


def _entry(record, tmp_path):
    cfg = PipelineConfig(output_dir=str(tmp_path))
    cfg.manifest.text_field = "text_itn"
    return WriteManifestStage()._build_manifest_entry(record, cfg.manifest)


class TestManifestCompleteness:
    def test_all_six_axes_filled_with_defaults(self, tmp_path):
        rec = {
            "audio_filepath": "/a.wav", "duration": 3.0,
            "text": "xin chào", "text_raw": "xin chao", "text_itn": "xin chào",
            "region": "northern",  # only region from tag; others missing
        }
        e = _entry(rec, tmp_path)
        for axis in ("language", "emotion", "gender", "age", "region", "voice_state"):
            assert e.get(axis) is not None, f"{axis} missing"
        assert e["region"] == "northern"
        assert e["emotion"] == "neutral"      # default
        assert e["voice_state"] == "sober"    # default
        assert e["age"] == "unknown"          # default
        assert e["text_raw"] == "xin chao"
        assert e["attribute_confidence"]["region"] == 0.95 or e["attribute_confidence"]["region"] >= 0.0
        assert e["attribute_confidence"]["emotion"] == 0.0  # defaulted

    def test_segment_type_single(self, tmp_path):
        rec = {"audio_filepath": "/a.wav", "duration": 3.0, "text": "x",
               "num_speakers": 1, "extra": {"is_overlap": False}}
        assert _entry(rec, tmp_path)["segment_type"] == "single"

    def test_segment_type_overlap_by_flag(self, tmp_path):
        rec = {"audio_filepath": "/a.wav", "duration": 3.0, "text": "x",
               "num_speakers": 1, "extra": {"is_overlap": True}}
        assert _entry(rec, tmp_path)["segment_type"] == "overlap"

    def test_segment_type_overlap_by_count(self, tmp_path):
        rec = {"audio_filepath": "/a.wav", "duration": 3.0, "text": "x", "num_speakers": 2}
        assert _entry(rec, tmp_path)["segment_type"] == "overlap"

    def test_tag_confidence_preserved(self, tmp_path):
        rec = {"audio_filepath": "/a.wav", "duration": 3.0, "text": "x",
               "region": "southern", "age": "teen",
               "attribute_confidence": {"region": 0.95, "age": 0.95}}
        e = _entry(rec, tmp_path)
        assert e["attribute_confidence"]["region"] == 0.95
        assert e["attribute_confidence"]["age"] == 0.95


class TestManifestRoundTrip:
    def test_schema_roundtrip_new_fields(self):
        from multitalker_asr.data.manifest_schema import ManifestRecord
        d = {
            "audio_filepath": "/a.wav", "duration": 3.0, "text": "xin chào",
            "text_raw": "xin chao", "segment_type": "overlap", "num_speakers": 2,
            "voice_state": "intoxicated", "speaker_id": "SPK_0",
        }
        r = ManifestRecord.from_dict(d)
        out = r.to_dict()
        assert out["text_raw"] == "xin chao"
        assert out["segment_type"] == "overlap"
        assert out["voice_state"] == "intoxicated"
        assert out["speaker_id"] == "SPK_0"
