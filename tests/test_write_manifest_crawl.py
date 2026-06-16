import json

import pytest

from multitalker_asr.data.pipeline.checkpoint import PipelineCheckpoint
from multitalker_asr.data.pipeline.config import PipelineConfig
from multitalker_asr.data.pipeline.stages.write_manifest import WriteManifestStage


def _seed(tmp_path):
    rows = [
        {"id": "v1", "task_id": "t1", "platform": "tiktok", "tags": ["STT Miền Bắc"],
         "url": "http://t/1", "channel": "c1", "data_type": "stt", "region": "northern"},
        {"id": "v2", "task_id": "t2", "platform": "youtube", "tags": ["TTS Vui"],
         "url": "http://y/2", "channel": "c2", "data_type": "tts", "emotion": "happy"},
        {"id": "v3", "task_id": "t3", "platform": "tiktok", "tags": ["STT Miền Nam"],
         "url": "http://t/3", "channel": "c3", "data_type": "stt", "region": "southern"},
    ]
    path = tmp_path / "seed.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(path)


def _records():
    # t1 → 2 kept segments, t2 → 1 kept segment, t3 → 0 kept (dropped upstream)
    return [
        {"id": "v1_s0", "audio_filepath": "/a/v1/s0.wav", "duration": 3.0, "text": "xin chào",
         "language": "vi", "gender": "female", "region": "northern", "consensus_score": 0.9,
         "platform": "tiktok", "extra": {"task_id": "t1", "platform": "tiktok"}},
        {"id": "v1_s1", "audio_filepath": "/a/v1/s1.wav", "duration": 2.0, "text": "các bạn",
         "language": "vi", "gender": "female", "region": "northern", "consensus_score": 0.8,
         "platform": "tiktok", "extra": {"task_id": "t1", "platform": "tiktok"}},
        {"id": "v2_s0", "audio_filepath": "/a/v2/s0.wav", "duration": 4.0, "text": "hôm nay vui",
         "language": "vi", "gender": "male", "emotion": "happy", "consensus_score": 0.85,
         "platform": "youtube", "extra": {"task_id": "t2", "platform": "youtube"}},
    ]


class TestWriteManifestCrawl:
    def test_shards_merged_and_ledger(self, tmp_path):
        cfg = PipelineConfig(output_dir=str(tmp_path))
        cfg.manifest.shard_by = "platform"
        cfg.crawl_seed.seed_path = _seed(tmp_path)
        ckpt = PipelineCheckpoint(checkpoint_dir=str(tmp_path / "ckpt"))

        WriteManifestStage().run(_records(), cfg, ckpt)

        tiktok = (tmp_path / "tiktok" / "manifest.jsonl").read_text().strip().splitlines()
        youtube = (tmp_path / "youtube" / "manifest.jsonl").read_text().strip().splitlines()
        merged = (tmp_path / "manifest_all.jsonl").read_text().strip().splitlines()
        assert len(tiktok) == 2
        assert len(youtube) == 1
        assert len(merged) == 3

        entry = json.loads(tiktok[0])
        assert entry["region"] == "northern"
        assert entry["language"] == "vi"
        assert entry["extra"]["platform"] == "tiktok"

    def test_dataset_ledger_lines_per_source(self, tmp_path):
        cfg = PipelineConfig(output_dir=str(tmp_path))
        cfg.manifest.shard_by = "platform"
        cfg.crawl_seed.seed_path = _seed(tmp_path)
        ckpt = PipelineCheckpoint(checkpoint_dir=str(tmp_path / "ckpt"))

        WriteManifestStage().run(_records(), cfg, ckpt)

        ledger_lines = (tmp_path / "dataset_metadata.jsonl").read_text().strip().splitlines()
        by_task = {json.loads(l)["task_id"]: json.loads(l) for l in ledger_lines}
        assert set(by_task) == {"t1", "t2", "t3"}
        assert by_task["t1"]["n_kept_segments"] == 2
        assert by_task["t1"]["kept_duration_s"] == pytest.approx(5.0)
        assert by_task["t2"]["n_kept_segments"] == 1
        assert by_task["t3"]["n_kept_segments"] == 0
        assert by_task["t3"]["status"] == "no_kept_segments"

        summary = json.loads((tmp_path / "dataset_summary.json").read_text())
        assert summary["n_source_items"] == 3
        assert summary["n_items_with_segments"] == 2
        assert summary["n_segments"] == 3
        assert summary["segments_by_platform"]["tiktok"] == 2
