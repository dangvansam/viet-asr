"""
CrawlStreamSource: stream completed tasks from the social-video-crawl DB (API),
download each task's audio + Vietnamese subtitle into a staging dir named by the
DB **task_id** (traceability), and build seed records + a per-task meta sidecar.

Source of truth is the crawl DB — every item keeps platform/tags/channel/url so a
segment can always be traced back to its task_id → DB row → original media on S3.
"""

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

from loguru import logger

from .client import CrawlAPIClient
from .minio_fetcher import MinIOFetcher
from .tag_taxonomy import parse_tag_labels

_WEAK_ATTR_KEYS = ("region", "age", "emotion", "voice_state")
_SUBTITLE_PREFERENCE = ("vi-orig", "vi", "vie-VN", "vi-VN", "vie", "vi-auto")


class CrawlStreamSource:
    def __init__(
        self,
        client: CrawlAPIClient,
        minio_fetcher: Optional[MinIOFetcher] = None,
        language: str = "vi",
        subtitle_langs: Optional[List[str]] = None,
    ):
        self._client = client
        self._minio = minio_fetcher
        self._language = language
        self._subtitle_langs = subtitle_langs or list(_SUBTITLE_PREFERENCE)

    def iter_tasks(
        self,
        platforms: Optional[List[str]] = None,
        tag_ids: Optional[List[int]] = None,
        status: str = "completed",
        page_limit: int = 1000,
        max_items: Optional[int] = None,
        skip_ids: Optional[Set[str]] = None,
    ) -> Iterator[Dict]:
        """Yield completed tasks that carry a media URL (audio or video), skipping
        already-done task_ids. Tasks with no playable media are dropped so every
        emitted item is guaranteed to yield audio after extraction."""
        skip = skip_ids or set()
        emitted = 0
        platform_list = platforms or [None]
        for platform in platform_list:
            for task in self._client.iter_tasks(platform, tag_ids, status, page_limit):
                tid = self.task_id(task)
                if not tid or tid in skip or not self.has_media(task):
                    continue
                yield task
                emitted += 1
                if max_items is not None and emitted >= max_items:
                    return

    def download(self, task: Dict, staging_dir: str) -> Dict:
        """Fetch the task's audio (+subtitle) into staging_dir keyed by task_id.

        The crawl preview URL's exact key is unreliable (stale bucket / filename),
        but the item folder is right: list the folder in the data bucket and grab
        `audio.wav` (guaranteed audio) + the best Vietnamese `.vtt` (optional).
        Raises if no audio object is found so every emitted item has audio.
        """
        tid = self.task_id(task)
        staging = Path(staging_dir)
        staging.mkdir(parents=True, exist_ok=True)

        if task.get("audio_key"):
            return self._download_by_keys(task, staging, tid)

        folder = self._item_folder(task)
        if folder is None or self._minio is None:
            raise ValueError(f"task {tid}: no resolvable media folder")
        keys = self._minio.list_prefix(folder)

        audio_key = self._pick_audio(keys)
        if audio_key is None:
            raise ValueError(f"task {tid}: no audio.wav under {folder}")
        audio_path = str(staging / f"{tid}.wav")
        self._minio.fetch_key(audio_key, audio_path)

        subtitle_path = None
        sub_key = self._pick_subtitle(keys)
        if sub_key:
            subtitle_path = str(staging / f"{tid}.vtt")
            try:
                self._minio.fetch_key(sub_key, subtitle_path)
            except Exception as exc:
                logger.warning(f"subtitle fetch failed for {tid}: {exc}")
                subtitle_path = None

        return {
            "task_id": tid,
            "audio_path": audio_path,
            "subtitle_path": subtitle_path,
            "media_kind": "audio",
        }

    def _download_by_keys(self, task: Dict, staging: Path, tid: str) -> Dict:
        """Fetch using the DB row's authoritative object keys (no folder listing)."""
        if self._minio is None:
            raise ValueError(f"task {tid}: MinIO unavailable for direct-key fetch")
        audio_path = str(staging / f"{tid}.wav")
        self._minio.fetch_key(task["audio_key"], audio_path)

        subtitle_path = None
        sub_key = self._pick_subtitle_key(task.get("subtitle_keys") or {})
        if sub_key:
            subtitle_path = str(staging / f"{tid}.vtt")
            try:
                self._minio.fetch_key(sub_key, subtitle_path)
            except Exception as exc:
                logger.warning(f"subtitle fetch failed for {tid}: {exc}")
                subtitle_path = None

        return {
            "task_id": tid,
            "audio_path": audio_path,
            "subtitle_path": subtitle_path,
            "media_kind": "audio",
        }

    def _pick_subtitle_key(self, subtitle_keys: Dict[str, str]) -> Optional[str]:
        """Best subtitle object key from a {lang: key} map: prefer Vietnamese."""
        if not subtitle_keys:
            return None
        for lang in self._subtitle_langs:
            if lang in subtitle_keys:
                return subtitle_keys[lang]
        for lang, key in subtitle_keys.items():
            if "vi" in lang.lower():
                return key
        return next(iter(subtitle_keys.values()), None)

    def _item_folder(self, task: Dict) -> Optional[str]:
        """Bucket folder prefix for the task's media (from the preview URL)."""
        if self._minio is None:
            return None
        preview = task.get("preview", {}) or {}
        url = preview.get("audio_url") or preview.get("video_url")
        if not url:
            return None
        key = self._minio.generic_key_from_url(url)
        if not key or "/" not in key:
            return None
        return key.rsplit("/", 1)[0] + "/"

    def has_media(self, task: Dict) -> bool:
        preview = task.get("preview", {}) or {}
        return bool(preview.get("audio_url") or preview.get("video_url"))

    @staticmethod
    def _pick_audio(keys: List[str]) -> Optional[str]:
        wavs = [k for k in keys if k.lower().endswith(".wav")]
        if not wavs:
            return None
        for k in wavs:
            if k.rsplit("/", 1)[-1].lower() == "audio.wav":
                return k
        return wavs[0]

    def build_seed(self, task: Dict, staged: Dict) -> Dict:
        tag_names = [t.get("name", "") for t in task.get("tags", [])]
        weak = parse_tag_labels(tag_names)
        seed: Dict = {
            "id": staged["task_id"],
            "audio_filepath": staged["audio_path"],
            "subtitle_path": staged.get("subtitle_path"),
            "platform": task.get("platform"),
            "tags": tag_names,
            "url": task.get("url"),
            "channel": task.get("channel_name"),
            "task_id": staged["task_id"],
            "data_type": weak.get("data_type"),
            "media_kind": staged.get("media_kind", "audio"),
        }
        if self._language:
            seed["language"] = self._language
        for key in _WEAK_ATTR_KEYS:
            if key in weak:
                seed[key] = weak[key]
        return seed

    def write_meta(self, task: Dict, meta_dir: str, outcome: Optional[Dict] = None) -> str:
        """Persist full task metadata + processing outcome for future debugging."""
        tid = self.task_id(task)
        meta_path = Path(meta_dir)
        meta_path.mkdir(parents=True, exist_ok=True)
        tag_names = [t.get("name", "") for t in task.get("tags", [])]
        meta = {
            "task_id": tid,
            "url": task.get("url"),
            "platform": task.get("platform"),
            "download_type": task.get("download_type"),
            "media_type": task.get("media_type"),
            "channel_id": task.get("channel_id"),
            "channel_name": task.get("channel_name"),
            "tags": tag_names,
            "weak_labels": parse_tag_labels(tag_names),
            "preview": task.get("preview", {}),
            "outcome": outcome or {},
        }
        out = meta_path / f"{tid}.json"
        out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(out)

    @staticmethod
    def task_id(task: Dict) -> str:
        return str(task.get("task_id") or "")

    def _pick_subtitle(self, keys: List[str]) -> Optional[str]:
        """Best `.vtt` key in the item folder: prefer Vietnamese variants."""
        vtts = [k for k in keys if k.lower().endswith(".vtt")]
        if not vtts:
            return None
        for hint in ("vi-orig", "vie-vn", "vi-vn", ".vi.", "vie", "vi"):
            for k in vtts:
                if hint in k.rsplit("/", 1)[-1].lower():
                    return k
        return vtts[0]
