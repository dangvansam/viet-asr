"""
Direct-Postgres client for the social-video-crawl DB (read-only).

Drop-in alternative to CrawlAPIClient: same `count` / `get_page` / `iter_tasks`
contract and the same task-dict shape, but reads the `downloads` table straight
from Postgres instead of paging the HTTP API. Source of truth is unchanged — the
DB row's `task_id` still keys every staged segment.

Two prerequisites for the host pipeline to reach the DB:
  - the crawl compose must publish the `db` port (it binds `db:5432` internally);
  - creds come from config or fall back to the crawl `.env` POSTGRES_* values.

Postgres stores the enum *names* (uppercase: 'COMPLETED', 'YOUTUBE'), so status /
platform filters are upper-cased on the way in and lower-cased on the way out to
match the API's `.value` shape. Object keys are taken from the row's authoritative
`audio_path` / `subtitle_paths` (the `download/` prefix stripped), so the consumer
can fetch from MinIO directly without folder listing.
"""

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from loguru import logger

_CRAWL_ENV = "/home/samdv/social-video-crawl/.env"
_BUCKET_PREFIX = "download/"


class CrawlDBClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str = "postgres",
        password: str = "",
        dbname: str = "social_media_downloader",
    ):
        self._dsn = dict(host=host, port=port, user=user, password=password, dbname=dbname)
        self._conn = None

    @classmethod
    def from_config(cls, cfg, env_path: str = _CRAWL_ENV) -> "CrawlDBClient":
        """Build from a CrawlSourceConfig, falling back to crawl .env POSTGRES_*."""
        env = cls._load_env(env_path)
        return cls(
            host=cfg.db_host or env.get("POSTGRES_HOST") or "localhost",
            port=int(cfg.db_port or env.get("POSTGRES_PORT") or 5432),
            user=cfg.db_user or env.get("POSTGRES_USER") or "postgres",
            password=cfg.db_password or env.get("POSTGRES_PASSWORD") or "",
            dbname=cfg.db_name or env.get("POSTGRES_DBNAME") or "postgres",
        )

    def count(
        self,
        platform: Optional[str] = None,
        tag_ids: Optional[List[int]] = None,
        status: str = "completed",
    ) -> int:
        where, params = self._filters(platform, tag_ids, status)
        sql = f"SELECT count(*) AS n FROM downloads d {where}"
        row = self._query(sql, params, fetch_one=True)
        return int(row["n"]) if row else 0

    def get_page(
        self,
        platform: Optional[str] = None,
        tag_ids: Optional[List[int]] = None,
        status: str = "completed",
        page: int = 1,
        limit: int = 1000,
    ) -> List[Dict]:
        where, params = self._filters(platform, tag_ids, status)
        params = {**params, "limit": limit, "offset": max(0, (page - 1) * limit)}
        sql = (
            "SELECT d.task_id, d.url, "
            "lower(d.platform::text) AS platform, "
            "lower(d.download_type::text) AS download_type, "
            "d.media_type, d.channel_id, d.channel_name, "
            "d.audio_path, d.video_path, d.subtitle_paths, "
            "COALESCE(("
            "  SELECT json_agg(json_build_object('id', t.id, 'name', t.name)) "
            "  FROM download_tag_links l JOIN tags t ON t.id = l.tag_id "
            "  WHERE l.download_task_id = d.task_id"
            "), '[]') AS tags "
            f"FROM downloads d {where} "
            "ORDER BY d.created_at ASC "
            "LIMIT %(limit)s OFFSET %(offset)s"
        )
        return [self._to_task(r) for r in self._query(sql, params)]

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

    def _filters(
        self, platform: Optional[str], tag_ids: Optional[List[int]], status: str
    ) -> tuple:
        clauses = ["d.audio_path IS NOT NULL"]
        params: Dict[str, object] = {}
        if status:
            clauses.append("d.status::text = %(status)s")
            params["status"] = status.upper()
        if platform:
            clauses.append("d.platform::text = %(platform)s")
            params["platform"] = platform.upper()
        if tag_ids:
            clauses.append(
                "EXISTS (SELECT 1 FROM download_tag_links l "
                "WHERE l.download_task_id = d.task_id AND l.tag_id = ANY(%(tag_ids)s))"
            )
            params["tag_ids"] = list(tag_ids)
        return "WHERE " + " AND ".join(clauses), params

    def _to_task(self, row: Dict) -> Dict:
        tags = row.get("tags") or []
        audio_key = self._object_key(row.get("audio_path"))
        video_key = self._object_key(row.get("video_path"))
        subtitle_keys = self._subtitle_keys(row.get("subtitle_paths"))
        preview: Dict[str, object] = {}
        if audio_key:
            preview["audio_url"] = audio_key
        if video_key:
            preview["video_url"] = video_key
        if subtitle_keys:
            preview["subtitles"] = dict(subtitle_keys)
        return {
            "task_id": row.get("task_id"),
            "url": row.get("url"),
            "platform": row.get("platform"),
            "download_type": row.get("download_type"),
            "media_type": row.get("media_type"),
            "channel_id": row.get("channel_id"),
            "channel_name": row.get("channel_name"),
            "tags": tags,
            "preview": preview,
            "audio_key": audio_key,
            "video_key": video_key,
            "subtitle_keys": subtitle_keys,
        }

    @staticmethod
    def _object_key(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        return path[len(_BUCKET_PREFIX):] if path.startswith(_BUCKET_PREFIX) else path

    @classmethod
    def _subtitle_keys(cls, raw: Optional[str]) -> Dict[str, str]:
        if not raw:
            return {}
        try:
            subs = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        keys: Dict[str, str] = {}
        for lang, path in (subs or {}).items():
            key = cls._object_key(path)
            if key:
                keys[lang] = key
        return keys

    def _query(self, sql: str, params: Dict, fetch_one: bool = False):
        from psycopg2.extras import RealDictCursor

        conn = self._ensure_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchone() if fetch_one else cur.fetchall()

    def _ensure_conn(self):
        if self._conn is None or self._conn.closed:
            import psycopg2

            logger.info(
                f"CrawlDBClient connecting host={self._dsn['host']}:{self._dsn['port']} "
                f"db={self._dsn['dbname']}"
            )
            self._conn = psycopg2.connect(**self._dsn)
            self._conn.set_session(readonly=True, autocommit=True)
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()

    @staticmethod
    def _load_env(env_path: str) -> Dict[str, str]:
        env: Dict[str, str] = {}
        path = Path(env_path)
        if not path.exists():
            return env
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                env.setdefault(key.strip(), val.strip())
        return env
