"""
SpeakerEmbedder: HTTP client to the speaker service `POST /v1/audio/embeddings`
(ECAPA-TDNN) — OpenAI-style audio contract (multipart `file` in; `{embedding, dim,
score, confidence}` out). Tolerant of the legacy `/embed` shape during migration.
"""

from typing import Optional

import numpy as np
from loguru import logger


class SpeakerEmbedder:
    def __init__(
        self,
        url: str = "http://localhost:2010",
        timeout: int = 30,
        verify: bool = False,
    ):
        # Accept a base URL or a legacy endpoint URL (…/embed) — derive the base.
        base = url.rstrip("/")
        for suffix in ("/v1/audio/embeddings", "/embed"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        self._base_url = base.rstrip("/")
        self._timeout = timeout
        self._verify = verify
        self._session = None

    def _ensure_session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def health(self) -> bool:
        import requests

        for path in ("/health", "/health_check", "/"):
            try:
                resp = requests.get(f"{self._base_url}{path}", timeout=5)
                if resp.status_code == 200:
                    return True
            except Exception:
                continue
        return False

    def embed(self, audio_path: str) -> Optional[np.ndarray]:
        """POST the audio to /v1/audio/embeddings; return the embedding vector or None.

        Falls back to the legacy `/embed` (field `voice_data`) when the server hasn't
        been upgraded yet (404) — zero-downtime rollout.
        """
        payload = self._request(
            audio_path,
            f"{self._base_url}/v1/audio/embeddings",
            "file",
            {"embedding": "true", "verify": str(self._verify).lower()},
        )
        if payload is None:
            payload = self._request(audio_path, f"{self._base_url}/embed", "voice_data", {})
        if payload is None:
            return None

        vec = self._extract_embedding(payload)
        if vec is None:
            logger.warning(f"embed: no embedding in response for {audio_path}")
            return None
        arr = np.asarray(vec, dtype=np.float32)
        # Service may return shape (W, 1, D) for W windows → mean to a (D,) vector.
        if arr.ndim > 1:
            arr = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
        return arr

    def _request(self, audio_path, url, field, data):
        """POST once; return JSON, None on 404 (caller may try legacy), None on error."""
        try:
            with open(audio_path, "rb") as f:
                resp = self._ensure_session().post(
                    url, files={field: f}, data=data, timeout=self._timeout
                )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning(f"embed error for {audio_path} at {url}: {exc}")
            return None

    @staticmethod
    def _extract_embedding(payload: dict):
        if not isinstance(payload, dict):
            return None
        if payload.get("embedding") is not None:  # OpenAI-style top-level
            return payload["embedding"]
        return (payload.get("data") or {}).get("embedding")  # legacy /embed shape

    def unload(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
