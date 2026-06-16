"""
MinIO SDK fetcher. Presigned VNG vStorage URLs 403 on object keys containing
spaces (TikTok channel ids), so we live-sign GETs via the MinIO SDK instead.
Derives the object key from the preview's presigned URL.
"""

import os
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

_CRAWL_ENV = "/home/samdv/social-video-crawl/.env"


class MinIOFetcher:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        region: Optional[str] = None,
        secure: bool = True,
    ):
        self._endpoint = endpoint
        self._bucket = bucket
        self._client = None
        self._params = dict(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            region=region,
            secure=secure,
        )

    @classmethod
    def from_params(
        cls,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        region: Optional[str] = None,
        secure: bool = True,
    ) -> "MinIOFetcher":
        return cls(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            bucket=bucket,
            region=region,
            secure=secure,
        )

    @classmethod
    def from_env(cls, env_path: str = _CRAWL_ENV) -> Optional["MinIOFetcher"]:
        env = cls._load_env(env_path)
        endpoint = env.get("MINIO_ENDPOINT")
        access = env.get("MINIO_ACCESS_KEY")
        secret = env.get("MINIO_SECRET_KEY")
        bucket = env.get("MINIO_BUCKET")
        if not (endpoint and access and secret and bucket):
            logger.warning("MinIO credentials incomplete — falling back to HTTP downloads")
            return None
        return cls(
            endpoint=endpoint,
            access_key=access,
            secret_key=secret,
            bucket=bucket,
            region=env.get("MINIO_REGION"),
        )

    def key_from_url(self, url: str) -> Optional[str]:
        marker = f"/{self._bucket}/"
        path = urllib.parse.urlparse(url).path
        if marker not in path:
            return None
        return urllib.parse.unquote(path.split(marker, 1)[1])

    def can_fetch(self, url: str) -> bool:
        return self._endpoint in url and self.key_from_url(url) is not None

    def fetch(self, url: str, dest_path: str) -> None:
        key = self.key_from_url(url)
        if key is None:
            raise ValueError(f"Cannot derive MinIO key from url: {url[:80]}")
        self.fetch_key(key, dest_path)

    def generic_key_from_url(self, url: str) -> Optional[str]:
        """Object key with the leading bucket segment stripped, regardless of the
        bucket name in the URL (crawl presigned URLs may name a stale bucket)."""
        path = urllib.parse.urlparse(url).path.lstrip("/")
        if "/" not in path:
            return None
        return urllib.parse.unquote(path.split("/", 1)[1])

    def list_prefix(self, prefix: str) -> List[str]:
        """Object keys under a prefix in the configured bucket."""
        return [
            obj.object_name
            for obj in self._ensure_client().list_objects(
                self._bucket, prefix=prefix, recursive=True
            )
        ]

    def fetch_key(self, key: str, dest_path: str) -> None:
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_client().fget_object(self._bucket, key, dest_path)

    def _ensure_client(self):
        if self._client is None:
            from minio import Minio

            self._client = Minio(**self._params)
        return self._client

    @staticmethod
    def _load_env(env_path: str) -> Dict[str, str]:
        env: Dict[str, str] = {}
        for key in ("MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MINIO_BUCKET", "MINIO_REGION"):
            if os.environ.get(key):
                env[key] = os.environ[key]
        path = Path(env_path)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env.setdefault(k.strip(), v.strip())
        return env
