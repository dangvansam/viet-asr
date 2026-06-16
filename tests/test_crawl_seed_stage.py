import json

import pytest

from multitalker_asr.data.pipeline.checkpoint import PipelineCheckpoint
from multitalker_asr.data.pipeline.config import PipelineConfig
from multitalker_asr.data.pipeline.stages.crawl_seed import CrawlSeedStage


def _write_seed(tmp_path, rows):
    path = tmp_path / "seed.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(path)


class TestCrawlSeedStage:
    def test_builds_records_with_weak_labels(self, tmp_path):
        seed = _write_seed(tmp_path, [
            {
                "id": "vid1", "audio_filepath": "/a/vid1/audio.wav",
                "subtitle_path": "/a/vid1/subtitle.vtt", "platform": "tiktok",
                "tags": ["STT Miền Bắc"], "url": "http://t/1", "channel": "ch",
                "task_id": "t1", "data_type": "stt", "language": "vi",
                "region": "northern",
            },
        ])
        cfg = PipelineConfig(output_dir=str(tmp_path))
        cfg.crawl_seed.seed_path = seed
        ckpt = PipelineCheckpoint(checkpoint_dir=str(tmp_path / "ckpt"))

        out = CrawlSeedStage().run([], cfg, ckpt)
        assert len(out) == 1
        rec = out[0]
        assert rec["id"] == "vid1"
        assert rec["audio_filepath"] == "/a/vid1/audio.wav"
        assert rec["language"] == "vi"
        assert rec["region"] == "northern"
        assert rec["platform"] == "tiktok"
        assert rec["extra"]["task_id"] == "t1"
        assert rec["extra"]["subtitle_path"] == "/a/vid1/subtitle.vtt"
        assert rec["extra"]["weak_labels"]["region"] == "northern"
        # tag-derived labels are stamped with confidence so FilterStage won't mask them
        assert rec["attribute_confidence"]["region"] >= 0.5
        assert rec["attribute_confidence"]["language"] >= 0.5

    def test_missing_seed_path_returns_input(self, tmp_path):
        cfg = PipelineConfig(output_dir=str(tmp_path))
        ckpt = PipelineCheckpoint(checkpoint_dir=str(tmp_path / "ckpt"))
        assert CrawlSeedStage().run([{"id": "x"}], cfg, ckpt) == [{"id": "x"}]

    def test_missing_file_raises(self, tmp_path):
        cfg = PipelineConfig(output_dir=str(tmp_path))
        cfg.crawl_seed.seed_path = str(tmp_path / "nope.jsonl")
        ckpt = PipelineCheckpoint(checkpoint_dir=str(tmp_path / "ckpt"))
        with pytest.raises(FileNotFoundError):
            CrawlSeedStage().run([], cfg, ckpt)
