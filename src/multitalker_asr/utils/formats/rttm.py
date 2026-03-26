import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union


@dataclass
class RTTMSegment:
    file_id: str
    channel: int
    start: float
    duration: float
    speaker: str

    @property
    def end(self) -> float:
        return self.start + self.duration

    def to_rttm_line(self) -> str:
        return (
            f"SPEAKER {self.file_id} {self.channel} "
            f"{self.start:.4f} {self.duration:.4f} "
            f"<NA> <NA> {self.speaker} <NA> <NA>"
        )


class RTTMConverter:
    @staticmethod
    def parse_line(line: str) -> Optional[RTTMSegment]:
        parts = line.strip().split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            return None

        return RTTMSegment(
            file_id=parts[1],
            channel=int(parts[2]),
            start=float(parts[3]),
            duration=float(parts[4]),
            speaker=parts[7],
        )

    @staticmethod
    def parse_file(rttm_path: Union[str, Path]) -> List[RTTMSegment]:
        segments = []
        path = Path(rttm_path)
        if not path.exists():
            return segments

        with open(path, "r") as f:
            for line in f:
                segment = RTTMConverter.parse_line(line)
                if segment is not None:
                    segments.append(segment)

        return segments

    @staticmethod
    def parse_lines(lines: List[str]) -> List[RTTMSegment]:
        segments = []
        for line in lines:
            segment = RTTMConverter.parse_line(line)
            if segment is not None:
                segments.append(segment)
        return segments

    @staticmethod
    def to_lines(segments: List[RTTMSegment]) -> List[str]:
        return [seg.to_rttm_line() for seg in segments]

    @staticmethod
    def write_file(
        segments: List[RTTMSegment],
        output_path: Union[str, Path],
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = RTTMConverter.to_lines(segments)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    @staticmethod
    def from_supervisions(
        supervisions,
        sample_id: str,
        channel: int = 1,
    ) -> List[RTTMSegment]:
        segments = []
        for seg in supervisions:
            segments.append(
                RTTMSegment(
                    file_id=sample_id,
                    channel=channel,
                    start=seg.start,
                    duration=seg.duration,
                    speaker=seg.speaker,
                )
            )
        return segments

    @staticmethod
    def get_speakers(segments: List[RTTMSegment]) -> List[str]:
        return list(set(seg.speaker for seg in segments))

    @staticmethod
    def filter_by_speaker(
        segments: List[RTTMSegment],
        speaker: str,
    ) -> List[RTTMSegment]:
        return [seg for seg in segments if seg.speaker == speaker]
