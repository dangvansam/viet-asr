from pathlib import Path

from multitalker_asr.data.pipeline.config import PipelineConfig
from multitalker_asr.data.pipeline.stream_runner import StreamPipeline


class FakeSource:
    """Stand-in for CrawlStreamSource: deterministic tasks, synthetic downloads."""

    def __init__(self, staging: Path, n_tasks: int = 4):
        self._staging = staging
        self._tasks = [
            {
                "task_id": f"task{i}",
                "platform": "tiktok" if i % 2 else "youtube",
                "tags": [{"name": "STT Miền Bắc"}],
                "url": f"https://x/{i}",
                "channel_name": f"chan{i}",
                "preview": {"audio_url": f"s3://b/{i}.wav"},
            }
            for i in range(n_tasks)
        ]

    def iter_tasks(self, platforms=None, tag_ids=None, status="completed",
                   page_limit=1000, max_items=None, skip_ids=None):
        skip = skip_ids or set()
        emitted = 0
        for t in self._tasks:
            if t["task_id"] in skip:
                continue
            yield t
            emitted += 1
            if max_items is not None and emitted >= max_items:
                return

    def download(self, task, staging_dir):
        tid = task["task_id"]
        staging = Path(staging_dir)
        staging.mkdir(parents=True, exist_ok=True)
        wav = staging / f"{tid}.wav"
        wav.write_bytes(b"\0" * 64)
        return {"task_id": tid, "audio_path": str(wav), "subtitle_path": None}

    def build_seed(self, task, staged):
        return {
            "id": staged["task_id"],
            "audio_filepath": staged["audio_path"],
            "task_id": staged["task_id"],
            "platform": task["platform"],
            "tags": [t["name"] for t in task["tags"]],
            "url": task["url"],
            "channel": task["channel_name"],
            "language": "vi",
        }

    def write_meta(self, task, meta_dir, outcome=None):
        tid = task["task_id"]
        md = Path(meta_dir)
        md.mkdir(parents=True, exist_ok=True)
        (md / f"{tid}.json").write_text("{}", encoding="utf-8")
        return str(md / f"{tid}.json")

    @staticmethod
    def task_id(task):
        return str(task["task_id"])


class FakePipeline:
    """Stand-in DataPipeline: 2 segments per record + simulated intermediates."""

    def __init__(self, out: Path):
        self._out = out
        (out / "segments").mkdir(parents=True, exist_ok=True)
        (out / "extracted").mkdir(parents=True, exist_ok=True)

    def run_records(self, records):
        segs = []
        for r in records:
            tid = r["id"]
            (self._out / "extracted" / f"{tid}.wav").write_bytes(b"\0" * 32)
            for k in range(2):
                seg_path = self._out / "segments" / f"{tid}_SPK_{k}.wav"
                seg_path.write_bytes(b"\0" * 16)
                seg = dict(r)
                seg.update({
                    "id": f"{tid}_SPK_{k}",
                    "audio_filepath": str(seg_path),
                    "text": "xin chào",
                    "duration": 2.0,
                    "num_speakers": 1,
                    "speaker_id": f"SPK_{k}",
                })
                seg["extra"] = {"task_id": tid, "platform": r.get("platform")}
                segs.append(seg)
        return segs


def _make_config(tmp_path, n_workers=2):
    cfg = PipelineConfig(output_dir=str(tmp_path), stages=[])
    cfg.stream.gpu_workers = n_workers
    cfg.stream.download_workers = 2
    cfg.stream.batch_size = 2
    cfg.stream.devices = ["cpu"]
    cfg.stream.queue_size = 8
    cfg.stream.manifest_flush_every = 1
    cfg.manifest.text_field = "text"
    cfg.manifest.shard_by = "platform"
    cfg.crawl_source.keep_meta = True
    return cfg


def _build(cfg, n_tasks=4):
    sp = StreamPipeline(cfg)
    sp._source = FakeSource(sp._staging, n_tasks=n_tasks)
    sp._make_worker_pipeline = lambda idx: FakePipeline(sp._out)
    return sp


class TestStreamPipeline:
    def test_parallel_accumulates_without_clobber(self, tmp_path):
        sp = _build(_make_config(tmp_path), n_tasks=4)
        summary = sp.run()

        assert summary["tasks"] == 4
        assert summary["segments"] == 8
        manifest = tmp_path / "manifest_all.jsonl"
        lines = [l for l in manifest.read_text().splitlines() if l.strip()]
        assert len(lines) == 8                         # 4 tasks x 2 segs, no clobber
        paths = [__import__("json").loads(l)["audio_filepath"] for l in lines]
        assert len(set(paths)) == 8                    # no duplicates across workers

    def test_keeps_segments_deletes_raw_and_extracted(self, tmp_path):
        sp = _build(_make_config(tmp_path), n_tasks=4)
        sp.run()

        assert len(list((tmp_path / "segments").glob("*.wav"))) == 8
        assert list((tmp_path / "_staging").glob("*.wav")) == []   # raw deleted
        assert list((tmp_path / "extracted").glob("*.wav")) == []  # intermediates deleted
        assert len(list((tmp_path / "meta").glob("*.json"))) == 4  # debug meta kept

    def test_resume_skips_processed(self, tmp_path):
        cfg = _make_config(tmp_path)
        _build(cfg, n_tasks=4).run()

        sp2 = _build(_make_config(tmp_path), n_tasks=4)   # same output_dir
        summary = sp2.run()
        assert summary["skipped_resume"] == 4
        assert summary["tasks"] == 0                       # nothing new processed
        lines = [l for l in (tmp_path / "manifest_all.jsonl").read_text().splitlines() if l.strip()]
        assert len(lines) == 8                             # unchanged, no duplication
