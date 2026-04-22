"""
ExtractAudioStage: convert video/audio to 16kHz mono WAV using ffmpeg.
"""

import subprocess
from pathlib import Path
from typing import Dict, List

from loguru import logger

from ..base_stage import BaseStage
from ..checkpoint import PipelineCheckpoint
from ..config import PipelineConfig


class ExtractAudioStage(BaseStage):
    """Convert any video/audio file to 16kHz mono WAV."""

    name = "extract_audio"

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        """
        Convert each record's source file to a 16kHz mono WAV.
        Skips files already converted (checkpoint).
        Returns records with updated audio_filepath pointing to WAV.
        """
        to_process, done = self._skip_processed(records, checkpoint)
        out_dir = Path(config.output_dir) / "extracted"
        out_dir.mkdir(parents=True, exist_ok=True)

        processed: List[Dict] = []
        for record in to_process:
            src = record.get("audio_filepath") or record.get("source_video", "")
            if not src:
                logger.warning(f"Record {record['id']} has no source file, skipping")
                processed.append(record)
                continue

            out_path = str(out_dir / f"{record['id']}.wav")
            try:
                duration = self._extract_one(src, out_path)
                record = dict(record)
                record["audio_filepath"] = out_path
                record["duration"] = duration
                checkpoint.mark_processed(record["id"], self.name)
                checkpoint.save_state()
                logger.info(f"Extracted: {record['id']} ({duration:.1f}s)")
            except RuntimeError:
                raise
            except subprocess.CalledProcessError as e:
                logger.error(f"ffmpeg failed for {src}: {e}")
            processed.append(record)

        return done + processed

    def _extract_one(self, src_path: str, out_path: str) -> float:
        """
        Run ffmpeg: src → 16kHz mono WAV. Returns duration in seconds.
        Raises RuntimeError if ffmpeg not found.
        Raises subprocess.CalledProcessError on ffmpeg failure.
        """
        # Check ffmpeg availability
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found — install with: apt install ffmpeg")

        # Convert to 16kHz mono WAV
        subprocess.run(
            [
                "ffmpeg",
                "-i", src_path,
                "-ar", "16000",
                "-ac", "1",
                "-y",
                out_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Get duration via ffprobe
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                out_path,
            ],
            capture_output=True,
            text=True,
        )
        try:
            return float(result.stdout.strip())
        except (ValueError, AttributeError):
            return 0.0
