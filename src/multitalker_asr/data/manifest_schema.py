from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from loguru import logger

from .manifest import ManifestReader, ManifestWriter

SCHEMA_VERSION = 2

REQUIRED_BASE_KEYS = ("audio_filepath", "duration")

ATTRIBUTE_KEYS = ("language", "emotion", "gender", "age", "region")

DEFAULT_TEXT = ""
DEFAULT_LABEL = ""
DEFAULT_NUM_SPEAKERS = 1


@dataclass
class AttributeConfidence:
    language: float = 0.0
    emotion: float = 0.0
    gender: float = 0.0
    age: float = 0.0
    region: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AttributeConfidence":
        if not data:
            return cls()
        return cls(
            language=float(data.get("language", 0.0)),
            emotion=float(data.get("emotion", 0.0)),
            gender=float(data.get("gender", 0.0)),
            age=float(data.get("age", 0.0)),
            region=float(data.get("region", 0.0)),
        )


@dataclass
class ManifestRecord:
    audio_filepath: str
    duration: float
    offset: float = 0.0
    text: str = DEFAULT_TEXT
    text_raw: Optional[str] = None      # un-normalized (spoken) form
    text_itn: Optional[str] = None      # normalized (ITN + PnC)
    label: str = DEFAULT_LABEL
    num_speakers: int = DEFAULT_NUM_SPEAKERS
    speaker_id: Optional[str] = None
    segment_type: str = "single"        # "single" | "overlap"
    language: Optional[str] = None
    emotion: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[str] = None
    region: Optional[str] = None
    voice_state: Optional[str] = None
    attribute_confidence: AttributeConfidence = field(default_factory=AttributeConfidence)
    source: Optional[str] = None
    schema_version: int = SCHEMA_VERSION
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "audio_filepath": self.audio_filepath,
            "offset": self.offset,
            "duration": self.duration,
            "text": self.text,
            "label": self.label,
            "num_speakers": self.num_speakers,
            "segment_type": self.segment_type,
            "schema_version": self.schema_version,
        }
        if self.text_raw is not None:
            out["text_raw"] = self.text_raw
        if self.text_itn is not None:
            out["text_itn"] = self.text_itn
        if self.speaker_id is not None:
            out["speaker_id"] = self.speaker_id
        if self.voice_state is not None:
            out["voice_state"] = self.voice_state
        for key in ATTRIBUTE_KEYS:
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if any(v > 0.0 for v in self.attribute_confidence.to_dict().values()):
            out["attribute_confidence"] = self.attribute_confidence.to_dict()
        if self.source is not None:
            out["source"] = self.source
        if self.extra:
            out["extra"] = self.extra
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManifestRecord":
        for key in REQUIRED_BASE_KEYS:
            if key not in data:
                raise ValueError(
                    f"Manifest entry missing required key '{key}': keys={list(data.keys())}"
                )

        known_keys = {
            "audio_filepath",
            "offset",
            "duration",
            "text",
            "text_raw",
            "text_itn",
            "label",
            "num_speakers",
            "speaker_id",
            "segment_type",
            "language",
            "emotion",
            "gender",
            "age",
            "region",
            "voice_state",
            "attribute_confidence",
            "source",
            "schema_version",
            "extra",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        return cls(
            audio_filepath=str(data["audio_filepath"]),
            duration=float(data["duration"]),
            offset=float(data.get("offset", 0.0)),
            text=str(data.get("text", DEFAULT_TEXT)),
            text_raw=data.get("text_raw"),
            text_itn=data.get("text_itn"),
            label=str(data.get("label", DEFAULT_LABEL)),
            num_speakers=int(data.get("num_speakers", DEFAULT_NUM_SPEAKERS)),
            speaker_id=data.get("speaker_id"),
            segment_type=str(data.get("segment_type", "single")),
            language=data.get("language"),
            emotion=data.get("emotion"),
            gender=data.get("gender"),
            age=data.get("age"),
            region=data.get("region"),
            voice_state=data.get("voice_state"),
            attribute_confidence=AttributeConfidence.from_dict(
                data.get("attribute_confidence")
            ),
            source=data.get("source"),
            schema_version=int(data.get("schema_version", 1)),
            extra={**extra, **(data.get("extra") or {})},
        )


class ManifestMigrator:
    def __init__(self, default_source: Optional[str] = None):
        self._default_source = default_source

    def upgrade_entry(self, data: Dict[str, Any]) -> Dict[str, Any]:
        record = ManifestRecord.from_dict(data)
        if record.source is None and self._default_source is not None:
            record.source = self._default_source
        record.schema_version = SCHEMA_VERSION
        return record.to_dict()

    def upgrade_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> int:
        reader = ManifestReader(input_path)
        writer = ManifestWriter(output_path)
        upgraded: List[Dict[str, Any]] = []
        for entry in reader:
            upgraded.append(self.upgrade_entry(entry))
        writer.write(upgraded)
        logger.success(f"Migrated {len(upgraded)} entries: {input_path} -> {output_path}")
        return len(upgraded)


class TypedManifestReader:
    def __init__(self, manifest_path: Union[str, Path], strict: bool = False):
        self._reader = ManifestReader(manifest_path)
        self._strict = strict

    def __iter__(self) -> Iterator[ManifestRecord]:
        for entry in self._reader:
            try:
                yield ManifestRecord.from_dict(entry)
            except ValueError as exc:
                if self._strict:
                    raise
                logger.warning(f"Skipping invalid manifest entry: {exc}")
                continue

    def read_all(self) -> List[ManifestRecord]:
        return list(self)


class TypedManifestWriter:
    def __init__(self, output_path: Union[str, Path]):
        self._writer = ManifestWriter(output_path)

    def write(self, records: List[ManifestRecord]) -> None:
        self._writer.write([r.to_dict() for r in records])

    def append(self, record: ManifestRecord) -> None:
        self._writer.append(record.to_dict())
