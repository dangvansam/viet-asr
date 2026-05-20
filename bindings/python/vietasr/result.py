# Keep annotations lazy so `dict[...]` / `list[...]` generics do not need
# evaluation at runtime — required to import on Python 3.8.
from __future__ import annotations

import json
from typing import Any


class Result:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    @classmethod
    def from_json(cls, raw: str) -> "Result":
        return cls(json.loads(raw or "{}"))

    @property
    def text(self) -> str:
        return self.payload.get("text", "")

    @property
    def partial(self) -> str:
        return self.payload.get("partial", "")

    @property
    def is_final(self) -> bool:
        return bool(self.payload.get("is_final", False))

    @property
    def segments(self) -> list[dict[str, Any]]:
        return self.payload.get("segments", [])

    @property
    def speakers(self) -> list[dict[str, Any]]:
        return self.payload.get("speakers", [])

    def field(self, key: str) -> Any:
        return self.payload.get(key)

    def to_json(self) -> str:
        return json.dumps(self.payload, ensure_ascii=False)

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def __contains__(self, key: str) -> bool:
        return key in self.payload

    def __repr__(self) -> str:
        return f"Result({self.payload!r})"
