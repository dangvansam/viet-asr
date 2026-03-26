import json
from enum import Enum
from pathlib import Path
from typing import Iterator, List, Optional, Union

from loguru import logger


class ManifestFormat(Enum):
    LHOTSE_CUTSET = "lhotse_cutset"
    NEMO_JSONL = "nemo_jsonl"
    JSON_ARRAY = "json_array"
    AUTO = "auto"


class TextExtractor:
    def __init__(self, manifest_format: ManifestFormat = ManifestFormat.AUTO):
        self._format = manifest_format

    def extract(
        self,
        manifest_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> List[str]:
        manifest_path = Path(manifest_path)
        texts = list(self._iter_texts(manifest_path))

        if output_path is not None:
            self._write_texts(texts, Path(output_path))
            logger.info(f"Extracted {len(texts)} utterances to {output_path}")

        return texts

    def extract_from_multiple(
        self,
        manifest_paths: List[Union[str, Path]],
        output_path: Optional[Union[str, Path]] = None,
        deduplicate: bool = True,
    ) -> List[str]:
        all_texts = []
        for path in manifest_paths:
            texts = self.extract(path)
            all_texts.extend(texts)

        if deduplicate:
            all_texts = list(set(all_texts))

        if output_path is not None:
            self._write_texts(all_texts, Path(output_path))
            logger.info(f"Extracted {len(all_texts)} unique utterances to {output_path}")

        return all_texts

    def _iter_texts(self, manifest_path: Path) -> Iterator[str]:
        if self._format == ManifestFormat.AUTO:
            yield from self._iter_texts_auto(manifest_path)
        elif self._format == ManifestFormat.LHOTSE_CUTSET:
            yield from self._iter_texts_lhotse(manifest_path)
        elif self._format == ManifestFormat.NEMO_JSONL:
            yield from self._iter_texts_jsonl(manifest_path)
        elif self._format == ManifestFormat.JSON_ARRAY:
            yield from self._iter_texts_json_array(manifest_path)

    def _iter_texts_auto(self, manifest_path: Path) -> Iterator[str]:
        try:
            yield from self._iter_texts_lhotse(manifest_path)
            return
        except Exception:
            pass

        try:
            yield from self._iter_texts_json_array(manifest_path)
            return
        except Exception:
            pass

        yield from self._iter_texts_jsonl(manifest_path)

    def _iter_texts_lhotse(self, manifest_path: Path) -> Iterator[str]:
        from lhotse import CutSet

        cuts = CutSet.from_file(manifest_path)
        for cut in cuts:
            for sup in cut.supervisions:
                if sup.text:
                    yield sup.text.strip()

    def _iter_texts_jsonl(self, manifest_path: Path) -> Iterator[str]:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    yield from self._extract_texts_from_dict(data)
                except json.JSONDecodeError:
                    continue

    def _iter_texts_json_array(self, manifest_path: Path) -> Iterator[str]:
        with open(manifest_path, "r", encoding="utf-8") as f:
            content = f.read()

        data = json.loads(content)

        if isinstance(data, list):
            for item in data:
                yield from self._extract_texts_from_dict(item)
        elif isinstance(data, dict):
            yield from self._extract_texts_from_dict(data)

    def _extract_texts_from_dict(self, data: dict) -> Iterator[str]:
        if "supervisions" in data:
            for sup in data["supervisions"]:
                if "text" in sup and sup["text"]:
                    yield sup["text"].strip()
        elif "text" in data and data["text"]:
            yield data["text"].strip()

    def _write_texts(self, texts: List[str], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + "\n")
