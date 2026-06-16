import argparse
import json
import re
import sys
from pathlib import Path

from loguru import logger

NON_LATIN_SCRIPTS = re.compile(
    "["
    "一-鿿"
    "぀-ゟ"
    "゠-ヿ"
    "가-힯"
    "฀-๿"
    "؀-ۿ"
    "Ѐ-ӿ"
    "ऀ-ॿ"
    "㐀-䶿"
    "]"
)


class SimpleManifestExporter:
    def __init__(self, language: str = "vi") -> None:
        self.language = language

    def is_vietnamese(self, record: dict) -> bool:
        if record.get("language") not in (None, self.language):
            return False
        text = record.get("text") or ""
        if not text.strip():
            return False
        if NON_LATIN_SCRIPTS.search(text):
            return False
        return True

    def to_timestamps(self, record: dict) -> list:
        timestamps = []
        for item in record.get("alignment") or []:
            timestamps.append(
                {
                    "word": item.get("text"),
                    "start": item.get("start_time"),
                    "end": item.get("end_time"),
                }
            )
        return timestamps

    def simplify(self, record: dict) -> dict:
        return {
            "audio_filepath": record.get("audio_filepath"),
            "duration": record.get("duration"),
            "text": record.get("text"),
            "timestamps": self.to_timestamps(record),
        }

    def run(self, input_path: Path, output_path: Path) -> None:
        total = 0
        kept = 0
        dropped_foreign = 0
        with input_path.open("r", encoding="utf-8") as reader, output_path.open(
            "w", encoding="utf-8"
        ) as writer:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                total += 1
                record = json.loads(line)
                if not self.is_vietnamese(record):
                    dropped_foreign += 1
                    continue
                writer.write(json.dumps(self.simplify(record), ensure_ascii=False) + "\n")
                kept += 1
        logger.success(
            f"Exported simple manifest total={total} kept={kept} "
            f"dropped_non_vietnamese={dropped_foreign} -> {output_path}"
        )


def parse_args(argv: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a manifest to Vietnamese-only and reduce it to the simple VAD+ASR schema."
    )
    parser.add_argument("--input", type=Path, default=Path("manifest_all.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("manifest_simple.jsonl"))
    parser.add_argument("--language", type=str, default="vi")
    return parser.parse_args(argv)


def main(argv: list) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        logger.error(f"Input manifest not found: {args.input}")
        return 1
    exporter = SimpleManifestExporter(language=args.language)
    exporter.run(args.input, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
