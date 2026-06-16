"""
WebVTT subtitle parser. Produces plain text and timed cues, stripping the
inline per-word timestamp tags (<00:00:00.520><c> word</c>) that yt-dlp emits
for auto-generated captions.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

_TIMESTAMP = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)
_INLINE_TAG = re.compile(r"<[^>]+>")
_HEADER_PREFIXES = ("WEBVTT", "Kind:", "Language:", "NOTE", "STYLE", "REGION")


@dataclass
class SubtitleCue:
    start: float
    end: float
    text: str


@dataclass
class SubtitleWord:
    word: str
    start: float
    end: float


class VTTParser:
    """Parse a .vtt file into timed cues and a deduplicated plain transcript."""

    def parse(self, path: Union[str, Path]) -> List[SubtitleCue]:
        path = Path(path)
        if not path.exists():
            return []

        cues: List[SubtitleCue] = []
        start = end = None
        buffer: List[str] = []

        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            match = _TIMESTAMP.search(line)
            if match:
                if start is not None:
                    self._flush(cues, start, end, buffer)
                start, end = self._parse_bounds(match)
                buffer = []
                continue
            if not line or line.startswith(_HEADER_PREFIXES) or line.isdigit():
                continue
            cleaned = _INLINE_TAG.sub("", line).strip()
            if cleaned:
                buffer.append(cleaned)

        if start is not None:
            self._flush(cues, start, end, buffer)
        return cues

    def text(self, path: Union[str, Path]) -> str:
        """Plain transcript: cue texts joined, consecutive duplicates removed."""
        lines: List[str] = []
        for cue in self.parse(path):
            if not lines or lines[-1] != cue.text:
                lines.append(cue.text)
        return " ".join(lines).strip()

    def words(
        self,
        path: Union[str, Path],
        start: float = None,
        end: float = None,
    ) -> List["SubtitleWord"]:
        """Per-word timings within [start, end] (whole file if bounds are None).

        VTT carries cue-level timing; split each cue's text into words and spread
        the cue span evenly so each word gets an approximate [start, end] — enough
        for time-aligned word voting against ASR word timings.
        """
        out: List[SubtitleWord] = []
        prev_text = None
        for cue in self.parse(path):
            if start is not None and cue.end <= float(start):
                continue
            if end is not None and cue.start >= float(end):
                continue
            if cue.text == prev_text:        # de-dup rolling auto-caption repeats
                continue
            prev_text = cue.text
            tokens = cue.text.split()
            if not tokens:
                continue
            span = max(cue.end - cue.start, 1e-3)
            step = span / len(tokens)
            for i, tok in enumerate(tokens):
                w_start = cue.start + i * step
                out.append(SubtitleWord(word=tok, start=w_start, end=w_start + step))
        return out

    def _flush(self, cues: List[SubtitleCue], start, end, buffer: List[str]) -> None:
        text = " ".join(buffer).strip()
        if text:
            cues.append(SubtitleCue(start=start, end=end, text=text))

    def _parse_bounds(self, match: re.Match):
        g = [int(x) for x in match.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0
        return start, end
