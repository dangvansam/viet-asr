"""
HTTP client for the social-video-crawl API (read-only).
Reads base URL + bearer token from env (CRAWL_API_URL, CRAWL_API_TOKEN).
"""

import os
from typing import Dict, Iterator, List, Optional

import requests
from loguru import logger


class CrawlAPIClient:
    DEFAULT_URL = "http://localhost:8010"

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: int = 30,
    ):
        self._base_url = (base_url or os.environ.get("CRAWL_API_URL") or self.DEFAULT_URL).rstrip("/")
        self._token = token or os.environ.get("CRAWL_API_TOKEN", "")
        self._timeout = timeout
        self._session = requests.Session()
        if self._token:
            self._session.headers["Authorization"] = f"Bearer {self._token}"

    def count(
        self,
        platform: Optional[str] = None,
        tag_ids: Optional[List[int]] = None,
        status: str = "completed",
    ) -> int:
        data = self._get_tasks(platform, tag_ids, status, page=1, limit=1)
        return int(data.get("total", 0))

    def get_page(
        self,
        platform: Optional[str] = None,
        tag_ids: Optional[List[int]] = None,
        status: str = "completed",
        page: int = 1,
        limit: int = 1000,
    ) -> List[Dict]:
        data = self._get_tasks(platform, tag_ids, status, page=page, limit=limit)
        return data.get("tasks", [])

    def iter_tasks(
        self,
        platform: Optional[str] = None,
        tag_ids: Optional[List[int]] = None,
        status: str = "completed",
        limit: int = 1000,
    ) -> Iterator[Dict]:
        page = 1
        while True:
            tasks = self.get_page(platform, tag_ids, status, page=page, limit=limit)
            if not tasks:
                break
            for task in tasks:
                yield task
            if len(tasks) < limit:
                break
            page += 1

    def resolve_file_url(self, url: str) -> str:
        """Presigned MinIO URLs are absolute; the fallback form is relative."""
        if url.startswith("http://") or url.startswith("https://"):
            return url
        return f"{self._base_url}/{url.lstrip('/')}"

    def _get_tasks(
        self,
        platform: Optional[str],
        tag_ids: Optional[List[int]],
        status: str,
        page: int,
        limit: int,
    ) -> Dict:
        params: Dict[str, object] = {
            "status": status,
            "page": page,
            "limit": limit,
            "sort_by": "created_at",
            "sort_order": "asc",
        }
        if platform:
            params["platform"] = platform
        if tag_ids:
            params["tag"] = ",".join(str(t) for t in tag_ids)
        resp = self._session.get(
            f"{self._base_url}/api/tasks", params=params, timeout=self._timeout
        )
        resp.raise_for_status()
        return resp.json()

    def download_to(self, url: str, dest_path: str) -> None:
        resolved = self.resolve_file_url(url)
        with self._session.get(resolved, stream=True, timeout=self._timeout, allow_redirects=True) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
