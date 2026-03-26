from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from .rttm import RTTMSegment


@dataclass
class AudacityLabel:
    start: float
    end: float
    label: str

    def to_line(self) -> str:
        return f"{self.start:.6f}\t{self.end:.6f}\t{self.label}"


class AudacityConverter:
    @staticmethod
    def from_rttm_segments(segments: List[RTTMSegment]) -> List[AudacityLabel]:
        labels = []
        for seg in segments:
            labels.append(
                AudacityLabel(
                    start=seg.start,
                    end=seg.end,
                    label=seg.speaker,
                )
            )
        return labels

    @staticmethod
    def from_rttm_lines(rttm_lines: List[str]) -> List[AudacityLabel]:
        from .rttm import RTTMConverter

        segments = RTTMConverter.parse_lines(rttm_lines)
        return AudacityConverter.from_rttm_segments(segments)

    @staticmethod
    def from_rttm_file(rttm_path: Union[str, Path]) -> List[AudacityLabel]:
        from .rttm import RTTMConverter

        segments = RTTMConverter.parse_file(rttm_path)
        return AudacityConverter.from_rttm_segments(segments)

    @staticmethod
    def from_supervisions(supervisions) -> List[AudacityLabel]:
        labels = []
        for seg in supervisions:
            end = seg.start + seg.duration
            labels.append(
                AudacityLabel(
                    start=seg.start,
                    end=end,
                    label=seg.speaker,
                )
            )
        return labels

    @staticmethod
    def to_lines(labels: List[AudacityLabel]) -> List[str]:
        return [label.to_line() for label in labels]

    @staticmethod
    def write_file(
        labels: List[AudacityLabel],
        output_path: Union[str, Path],
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = AudacityConverter.to_lines(labels)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    @staticmethod
    def parse_file(label_path: Union[str, Path]) -> List[AudacityLabel]:
        labels = []
        path = Path(label_path)
        if not path.exists():
            return labels

        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    labels.append(
                        AudacityLabel(
                            start=float(parts[0]),
                            end=float(parts[1]),
                            label=parts[2],
                        )
                    )

        return labels
