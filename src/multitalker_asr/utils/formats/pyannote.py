from pathlib import Path
from typing import List, Optional, Union

from .rttm import RTTMConverter, RTTMSegment


class PyannoteConverter:
    @staticmethod
    def rttm_to_annotation(
        rttm_path: Union[str, Path],
        uri: str = "",
    ):
        from pyannote.core import Annotation, Segment

        annotation = Annotation(uri=uri)
        path = Path(rttm_path)

        if not path.exists():
            return annotation

        segments = RTTMConverter.parse_file(path)
        for seg in segments:
            if seg.duration > 0:
                annotation[Segment(seg.start, seg.end)] = seg.speaker

        return annotation

    @staticmethod
    def segments_to_annotation(
        segments: List[RTTMSegment],
        uri: str = "",
    ):
        from pyannote.core import Annotation, Segment

        annotation = Annotation(uri=uri)
        for seg in segments:
            if seg.duration > 0:
                annotation[Segment(seg.start, seg.end)] = seg.speaker

        return annotation

    @staticmethod
    def annotation_to_segments(
        annotation,
        file_id: str = "unknown",
        channel: int = 1,
    ) -> List[RTTMSegment]:
        segments = []
        for segment, _, label in annotation.itertracks(yield_label=True):
            segments.append(
                RTTMSegment(
                    file_id=file_id,
                    channel=channel,
                    start=segment.start,
                    duration=segment.end - segment.start,
                    speaker=label,
                )
            )
        return segments

    @staticmethod
    def get_speakers_from_annotation(annotation) -> List[str]:
        return list(annotation.labels())

    @staticmethod
    def get_total_duration(annotation) -> float:
        from pyannote.core import Timeline

        timeline = annotation.get_timeline()
        return timeline.duration()
