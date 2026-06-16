import json

import pytest

from multitalker_asr.data.crawl import (
    CrawlAPIClient,
    MediaDownloader,
    StratifiedSampler,
    parse_tag_labels,
    primary_tag_slug,
    resolve_tag_family,
    slugify,
)
from multitalker_asr.data.crawl.tag_taxonomy import STT_TAG_IDS, TTS_TAG_IDS


class TestTagTaxonomy:
    def test_region_age_voice_emotion(self):
        assert parse_tag_labels(["STT Miền Bắc"]) == {"data_type": "stt", "region": "northern"}
        assert parse_tag_labels(["STT Miền Nam"])["region"] == "southern"
        assert parse_tag_labels(["STT Miền Trung"])["region"] == "central"
        assert parse_tag_labels(["TTS Trẻ em"]) == {"data_type": "tts", "age": "child"}
        assert parse_tag_labels(["STT Người già"])["age"] == "senior"
        assert parse_tag_labels(["STT Say rượu không tỉnh táo"])["voice_state"] == "intoxicated"
        assert parse_tag_labels(["TTS Buồn"])["emotion"] == "sad"
        assert parse_tag_labels(["TTS Vui"])["emotion"] == "happy"

    def test_no_match(self):
        assert parse_tag_labels(["Random Tag"]) == {}

    def test_primary_tag_slug_prefers_stt_tts(self):
        assert primary_tag_slug(["BTV VTV", "STT Miền Bắc"]) == "stt_mien_bac"
        assert primary_tag_slug([]) == "untagged"

    def test_slugify_strips_accents(self):
        assert slugify("STT Miền Bắc") == "stt_mien_bac"

    def test_resolve_tag_family(self):
        assert resolve_tag_family(["stt"]) == STT_TAG_IDS
        assert resolve_tag_family(["tts"]) == TTS_TAG_IDS
        assert len(resolve_tag_family(["stt", "tts"])) == len(STT_TAG_IDS) + len(TTS_TAG_IDS)


class _FakeClient:
    """Stand-in for CrawlAPIClient backed by an in-memory population."""

    def __init__(self, population):
        self._pop = population  # {(platform, tag_id): [task,...]}

    def count(self, platform=None, tag_ids=None, status="completed"):
        return len(self._pop.get((platform, tag_ids[0]), []))

    def get_page(self, platform=None, tag_ids=None, status="completed", page=1, limit=1000):
        items = self._pop.get((platform, tag_ids[0]), [])
        start = (page - 1) * limit
        return items[start : start + limit]


def _mk_tasks(platform, tag_id, n):
    return [
        {"task_id": f"{platform}-{tag_id}-{i}", "platform": platform,
         "tags": [{"id": tag_id, "name": "STT Miền Bắc"}], "preview": {}}
        for i in range(n)
    ]


class TestStratifiedSampler:
    def test_covers_every_cell_and_dedups(self):
        pop = {
            ("tiktok", 14): _mk_tasks("tiktok", 14, 500),
            ("youtube", 14): _mk_tasks("youtube", 14, 100),
            ("facebook", 14): _mk_tasks("facebook", 14, 3),
        }
        client = _FakeClient(pop)
        sampler = StratifiedSampler(page_limit=50, floor=1)
        result = sampler.sample(client, ["tiktok", "youtube", "facebook"], [14], total=60, seed=7)
        ids = [t["task_id"] for t in result.tasks]
        assert len(ids) == len(set(ids))  # deduped
        assert 1 <= result.realized["facebook:14"] <= 3  # floor guarantees coverage of tiny cell
        assert all(p in result.realized for p in ("tiktok:14", "youtube:14", "facebook:14"))
        assert len(result.tasks) == 60

    def test_deterministic_with_seed(self):
        pop = {("tiktok", 14): _mk_tasks("tiktok", 14, 200)}
        client = _FakeClient(pop)
        s = StratifiedSampler(page_limit=50)
        a = s.sample(client, ["tiktok"], [14], total=20, seed=1)
        b = s.sample(client, ["tiktok"], [14], total=20, seed=1)
        assert [t["task_id"] for t in a.tasks] == [t["task_id"] for t in b.tasks]

    def test_quota_capped_by_population(self):
        pop = {("tiktok", 14): _mk_tasks("tiktok", 14, 5)}
        client = _FakeClient(pop)
        s = StratifiedSampler(page_limit=50)
        result = s.sample(client, ["tiktok"], [14], total=100, seed=1)
        assert len(result.tasks) == 5


class TestMediaDownloader:
    def test_item_dir_and_meta(self, tmp_path):
        task = {
            "task_id": "abc", "platform": "tiktok",
            "tags": [{"id": 14, "name": "STT Miền Bắc"}],
            "preview": {"video_id": "vid123", "audio_url": None, "subtitles": {}},
        }
        dl = MediaDownloader(client=None, out_root=str(tmp_path))
        d = dl.item_dir(task)
        assert d == tmp_path / "tiktok" / "stt_mien_bac" / "vid123"
        result = dl.download(task, download_media=False)
        assert result["ok"] is True
        meta = json.loads((d / "meta.json").read_text())
        assert meta["weak_labels"] == {"data_type": "stt", "region": "northern"}

    def test_skip_existing(self, tmp_path):
        class _C:
            def __init__(self):
                self.calls = 0

            def download_to(self, url, dest):
                self.calls += 1
                open(dest, "wb").close()

        client = _C()
        task = {
            "task_id": "abc", "platform": "tiktok",
            "tags": [{"id": 14, "name": "STT Miền Bắc"}],
            "preview": {"video_id": "v", "audio_url": "http://x/a.wav", "subtitles": {}},
        }
        dl = MediaDownloader(client=client, out_root=str(tmp_path))
        dl.download(task)
        dl.download(task)
        assert client.calls == 1  # second call skips existing audio
