"""
Map social-video-crawl STT/TTS tag names to weak attribute labels.

The crawl taxonomy already encodes region/age/voice_state/emotion in tag names
(e.g. "STT Miền Bắc", "TTS Trẻ em"). These become weak labels used to seed the
pipeline and to cross-check against model predictions in ConsensusStage.
Label values are constrained to the model's attribute vocabulary
(see configs/attribute_vocab.py).
"""

import re
import unicodedata
from typing import Dict, List, Optional

STT_TAG_IDS: List[int] = [91, 17, 14, 16, 15, 18, 88, 34, 31, 33, 35, 36]
TTS_TAG_IDS: List[int] = [13, 39, 32, 38, 41, 37, 42, 40]
CRAWL_TAG_IDS: List[int] = STT_TAG_IDS + TTS_TAG_IDS


def resolve_tag_family(families: List[str]) -> List[int]:
    """Map family names ('stt'/'tts') to their tag-id lists (deduped)."""
    ids: List[int] = []
    for fam in families:
        key = fam.strip().lower()
        if key == "stt":
            ids.extend(STT_TAG_IDS)
        elif key == "tts":
            ids.extend(TTS_TAG_IDS)
    return list(dict.fromkeys(ids)) or list(CRAWL_TAG_IDS)

REGION_RULES = {
    "miền bắc": "northern",
    "miền trung": "central",
    "miền nam": "southern",
}

AGE_RULES = {
    "trẻ em": "child",
    "thanh niên": "teen",
    "trung niên": "adult",
    "người già": "senior",
}

VOICE_STATE_RULES = {
    "say rượu": "intoxicated",
}

EMOTION_RULES = {
    "buồn": "sad",
    "vui": "happy",
    "bực": "angry",
    "sợ": "fear",
    "nghiêm túc": "neutral",
}


def parse_tag_labels(tag_names: List[str]) -> Dict[str, str]:
    """Infer weak attribute labels from a list of crawl tag names.

    Only keys that can be confidently inferred are included. data_type is
    derived from the STT/TTS prefix.
    """
    joined = " ".join(tag_names).lower()
    labels: Dict[str, str] = {}

    data_type = _data_type(tag_names)
    if data_type:
        labels["data_type"] = data_type

    region = _first_match(joined, REGION_RULES)
    if region:
        labels["region"] = region

    age = _first_match(joined, AGE_RULES)
    if age:
        labels["age"] = age

    voice_state = _first_match(joined, VOICE_STATE_RULES)
    if voice_state:
        labels["voice_state"] = voice_state

    emotion = _first_match(joined, EMOTION_RULES)
    if emotion:
        labels["emotion"] = emotion

    return labels


def _data_type(tag_names: List[str]) -> Optional[str]:
    for name in tag_names:
        head = name.strip().lower()
        if head.startswith("stt"):
            return "stt"
        if head.startswith("tts"):
            return "tts"
    return None


def _first_match(text: str, rules: Dict[str, str]) -> Optional[str]:
    for needle, label in rules.items():
        if needle in text:
            return label
    return None


def primary_tag_slug(tag_names: List[str]) -> str:
    """Filesystem-safe folder name from the first STT/TTS tag (else first tag)."""
    if not tag_names:
        return "untagged"
    chosen = None
    for name in tag_names:
        head = name.strip().lower()
        if head.startswith("stt") or head.startswith("tts"):
            chosen = name
            break
    return slugify(chosen or tag_names[0])


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"[^\w\s-]", "", ascii_text).strip().lower()
    slug = re.sub(r"[\s_-]+", "_", ascii_text)
    return slug or "untagged"
