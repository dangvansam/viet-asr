"""
Abstract base class for all pipeline stages.

Every stage in the data processing pipeline inherits from BaseStage.
The `run()` method receives all records, filters out already-processed ones
via the checkpoint, processes the remaining, and returns the full set
(done + newly processed).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from .checkpoint import PipelineCheckpoint
from .config import PipelineConfig


class BaseStage(ABC):
    """Abstract interface that every pipeline stage must implement."""

    # Subclasses MUST set this to match the stage key in the YAML stages list
    name: str

    @abstractmethod
    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        """
        Process records, skip already-processed ones, return enriched records.

        Args:
            records: List of record dicts. Minimum keys: {"id": str, "audio_filepath": str}
            config: Full pipeline config (access stage sub-config via config.vad, etc.)
            checkpoint: Crash-resumable checkpoint manager

        Returns:
            List of processed record dicts (done + newly processed).
            Stages may expand records (1 input → N outputs, e.g. VAD diarization).
        """
        ...

    def _skip_processed(
        self,
        records: List[Dict],
        checkpoint: PipelineCheckpoint,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Split records into those requiring processing and those already done.

        Args:
            records: Full list of records
            checkpoint: Checkpoint manager

        Returns:
            (to_process, already_done) — two disjoint sublists of records
        """
        to_process: List[Dict] = []
        done: List[Dict] = []
        for r in records:
            if checkpoint.is_processed(r["id"], self.name):
                done.append(r)
            else:
                to_process.append(r)
        return to_process, done
