import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from loguru import logger


class ManifestReader:
    def __init__(self, manifest_path: Union[str, Path]):
        self._path = Path(manifest_path)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def read_all(self) -> List[Dict[str, Any]]:
        return list(self)

    def count(self) -> int:
        return sum(1 for _ in self)


class ManifestWriter:
    def __init__(self, output_path: Union[str, Path]):
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, entries: List[Dict[str, Any]]) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(entries)} entries to {self._path}")

    def append(self, entry: Dict[str, Any]) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def from_csv(
        input_csv: Union[str, Path],
        output_json: Union[str, Path],
        audio_dir: Union[str, Path],
    ) -> int:
        input_csv = Path(input_csv)
        output_json = Path(output_json)
        audio_dir = Path(audio_dir)
        manifest_data = []

        if not input_csv.exists():
            logger.warning(f"Input CSV {input_csv} not found. Creating dummy entry.")
            dummy_entry = {
                "audio_filepath": str(audio_dir / "demo_audio.wav"),
                "offset": 0.0,
                "duration": 2.5,
                "label": "speaker_0",
                "text": "demo text",
                "num_speakers": 1,
            }
            manifest_data.append(dummy_entry)
        else:
            with open(input_csv, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines[1:]:
                parts = line.strip().split(",", 4)
                if len(parts) >= 5:
                    filename, spk_id, start, dur, text = parts
                    audio_path = audio_dir / filename

                    if audio_path.exists():
                        entry = {
                            "audio_filepath": str(audio_path.absolute()),
                            "offset": float(start),
                            "duration": float(dur),
                            "label": spk_id,
                            "text": text,
                        }
                        manifest_data.append(entry)
                    else:
                        logger.warning(f"Audio file not found: {audio_path}")

        writer = ManifestWriter(output_json)
        writer.write(manifest_data)
        return len(manifest_data)
