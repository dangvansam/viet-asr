"""
Download crawl media + subtitle for a sampled task into a
{platform}/{primary_tag}/{video_id}/ folder, with a meta.json sidecar.
Resumable: skips files that already exist.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from .client import CrawlAPIClient
from .minio_fetcher import MinIOFetcher
from .tag_taxonomy import parse_tag_labels, primary_tag_slug

_SUBTITLE_PREFERENCE = ("vi-orig", "vi", "vie-VN", "vi-VN", "vie", "vi-auto")


class MediaDownloader:
    def __init__(
        self,
        client: CrawlAPIClient,
        out_root: str,
        subtitle_langs: Optional[List[str]] = None,
        minio_fetcher: Optional[MinIOFetcher] = None,
    ):
        self._client = client
        self._out_root = Path(out_root)
        self._subtitle_langs = subtitle_langs or list(_SUBTITLE_PREFERENCE)
        self._minio = minio_fetcher

    def _download_url(self, url: str, dest: str) -> None:
        if self._minio is not None and self._minio.can_fetch(url):
            self._minio.fetch(url, dest)
        else:
            self._client.download_to(url, dest)

    def item_dir(self, task: Dict) -> Path:
        platform = task.get("platform", "unknown")
        tag_names = [t.get("name", "") for t in task.get("tags", [])]
        ident = self._identifier(task)
        return self._out_root / platform / primary_tag_slug(tag_names) / ident

    def download(self, task: Dict, download_media: bool = True) -> Dict:
        out_dir = self.item_dir(task)
        out_dir.mkdir(parents=True, exist_ok=True)
        preview = task.get("preview", {}) or {}

        result: Dict = {
            "audio_path": None,
            "subtitle_path": None,
            "media_kind": None,
            "ok": False,
            "error": None,
        }

        try:
            if download_media:
                audio_path, kind = self._fetch_media(preview, out_dir)
                result["audio_path"] = audio_path
                result["media_kind"] = kind
                result["subtitle_path"] = self._fetch_subtitle(preview, out_dir)
            self._write_meta(task, out_dir)
            result["ok"] = True
        except Exception as exc:
            logger.error(f"Download failed for {self._identifier(task)}: {exc}")
            result["error"] = str(exc)

        return result

    def _fetch_media(self, preview: Dict, out_dir: Path):
        audio_url = preview.get("audio_url")
        if audio_url:
            dest = out_dir / "audio.wav"
            if not dest.exists():
                self._download_url(audio_url, str(dest))
            return str(dest), "audio"
        video_url = preview.get("video_url")
        if video_url:
            dest = out_dir / "video.mp4"
            if not dest.exists():
                self._download_url(video_url, str(dest))
            return str(dest), "video"
        return None, None

    def _fetch_subtitle(self, preview: Dict, out_dir: Path) -> Optional[str]:
        subtitles = preview.get("subtitles") or {}
        if not subtitles:
            return None
        lang = self._pick_subtitle_lang(subtitles)
        if lang is None:
            return None
        dest = out_dir / "subtitle.vtt"
        if not dest.exists():
            self._download_url(subtitles[lang], str(dest))
        return str(dest)

    def _pick_subtitle_lang(self, subtitles: Dict[str, str]) -> Optional[str]:
        for lang in self._subtitle_langs:
            if lang in subtitles:
                return lang
        for lang in subtitles:
            if lang.lower().startswith("vi"):
                return lang
        return next(iter(subtitles), None)

    def _write_meta(self, task: Dict, out_dir: Path) -> None:
        tag_names = [t.get("name", "") for t in task.get("tags", [])]
        meta = {
            "task_id": task.get("task_id"),
            "url": task.get("url"),
            "platform": task.get("platform"),
            "download_type": task.get("download_type"),
            "media_type": task.get("media_type"),
            "channel_id": task.get("channel_id"),
            "channel_name": task.get("channel_name"),
            "tags": tag_names,
            "weak_labels": parse_tag_labels(tag_names),
            "preview": task.get("preview", {}),
        }
        (out_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _identifier(self, task: Dict) -> str:
        preview = task.get("preview", {}) or {}
        return str(preview.get("video_id") or task.get("task_id") or "unknown")
