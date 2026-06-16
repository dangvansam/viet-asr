"""
Pure helpers shared by the LitServe audio micro-services (see `serving/lit_api.py`).

The hand-rolled FastAPI route factory + ServiceBatcher that used to live here were
retired when the services moved to LitServe; only the request/response shaping
helpers remain. They have no FastAPI/uvicorn import (stdlib + loguru only), so the
lightweight torch-free pipeline image can import them when needed.
"""

import math
from typing import Any, Dict, List, Optional


def parse_openai_form(form) -> Dict[str, Any]:
    """Normalize a multipart form (Starlette FormData) into OpenAI-style params.

    Collapses repeated keys (`timestamp_granularities[]`, `include[]`) into lists and
    coerces the boolean-ish flags (`stream`, `diarization`). The uploaded file field
    (any value with a `read` attribute) is skipped.
    """
    params: Dict[str, Any] = {}
    multi: Dict[str, List[str]] = {}
    for key, value in form.multi_items():
        if hasattr(value, "read"):
            continue
        base = key[:-2] if key.endswith("[]") else key
        if key.endswith("[]") or base in ("timestamp_granularities", "include"):
            multi.setdefault(base, []).append(value)
        else:
            params[base] = value
    params.update(multi)
    for flag in ("stream", "diarization"):
        if flag in params and not isinstance(params[flag], bool):
            params[flag] = str(params[flag]).lower() in ("1", "true", "yes")
    return params


def asr_result_to_verbose_json(
    result: "ASRResultLike",
    granularities: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
    model_id: str = "",
) -> dict:
    """Serialize an ASRResult into an OpenAI `verbose_json` transcription payload."""
    granularities = granularities or ["segment"]
    include = include or []
    want_words = "word" in granularities
    want_logprobs = "logprobs" in include

    timings = list(getattr(result, "word_timings", None) or [])
    words: List[dict] = []
    for w in timings:
        item: Dict[str, Any] = {
            "word": getattr(w, "word", ""),
            "start": float(getattr(w, "start", 0.0)),
            "end": float(getattr(w, "end", 0.0)),
        }
        conf = getattr(w, "confidence", None)
        if want_logprobs and conf is not None and conf > 0:
            item["logprob"] = math.log(min(max(conf, 1e-9), 1.0))
        words.append(item)

    text = getattr(result, "text", "") or ""
    duration = words[-1]["end"] if words else 0.0
    segment: Dict[str, Any] = {
        "id": 0,
        "start": words[0]["start"] if words else 0.0,
        "end": duration,
        "text": text,
    }
    confidence = float(getattr(result, "confidence", 1.0) or 1.0)
    if want_logprobs and 0 < confidence <= 1.0:
        segment["avg_logprob"] = math.log(confidence)
    if want_words:
        segment["words"] = words

    payload: Dict[str, Any] = {
        "task": "transcribe",
        "language": getattr(result, "language", None) or "",
        "duration": duration,
        "text": text,
        "segments": [segment] if (text or words) else [],
    }
    if want_words:
        payload["words"] = words
    if model_id:
        payload["model"] = model_id
    return payload
