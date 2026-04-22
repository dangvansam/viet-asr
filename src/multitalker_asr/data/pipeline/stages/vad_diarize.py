"""
VADDiarizeStage: speaker diarization via pyannote, clips segments to WAV files.
"""

import gc
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger

from ..base_stage import BaseStage
from ..checkpoint import PipelineCheckpoint
from ..config import PipelineConfig, VADConfig


class VADDiarizeStage(BaseStage):
    """Run speaker diarization on long WAV files and clip speaker segments."""

    name = "vad_diarize"

    def __init__(self) -> None:
        self._pipeline = None

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        """
        Run speaker diarization on long WAV files.
        Clips each speaker turn to a separate WAV file.
        Filters clips by min/max duration.
        Returns one record per accepted segment.
        """
        to_process, done = self._skip_processed(records, checkpoint)
        if not to_process:
            return done

        if not config.vad.hf_token:
            logger.warning("HF_TOKEN not set — pyannote may fail for first download")

        self._pipeline = self._load_pipeline(config.vad, config.device)

        seg_dir = Path(config.output_dir) / "segments"
        seg_dir.mkdir(parents=True, exist_ok=True)

        all_segments: List[Dict] = []

        try:
            import torchaudio

            for record in to_process:
                audio_path = record["audio_filepath"]
                try:
                    diarization = self._pipeline(audio_path)
                    waveform, sample_rate = torchaudio.load(audio_path)  # [1, N]

                    file_segments: List[Dict] = []
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        dur = turn.end - turn.start
                        if dur < config.vad.min_duration or dur > config.vad.max_duration:
                            continue

                        seg_id = self._make_segment_id(record["id"], speaker, turn.start, turn.end)
                        out_path = str(seg_dir / f"{seg_id}.wav")
                        self._clip_segment(waveform, sample_rate, turn.start, turn.end, out_path)

                        seg_record = {
                            "id": seg_id,
                            "audio_filepath": out_path,
                            "source_audio": audio_path,
                            "start": turn.start,
                            "end": turn.end,
                            "speaker_id": speaker,
                            "duration": dur,
                        }
                        file_segments.append(seg_record)

                    if not file_segments:
                        logger.warning(f"No segments kept from {audio_path}")

                    all_segments.extend(file_segments)
                    checkpoint.mark_processed(record["id"], self.name)
                    checkpoint.save_state()
                    logger.info(f"Diarized {record['id']}: {len(file_segments)} segments")

                    # Free waveform tensor
                    del waveform

                except Exception as e:
                    logger.error(f"Diarization failed for {record['id']}: {e}")
        finally:
            if self._pipeline is not None:
                del self._pipeline
                self._pipeline = None
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        return done + all_segments

    def _load_pipeline(self, vad_config: VADConfig, device: str):
        """Load pyannote diarization pipeline."""
        from pyannote.audio import Pipeline

        kwargs = {}
        if vad_config.hf_token:
            kwargs["use_auth_token"] = vad_config.hf_token

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            **kwargs,
        )
        pipeline.to(device)
        return pipeline

    def _clip_segment(
        self,
        waveform,  # torch.Tensor [1, N]
        sample_rate: int,
        start: float,
        end: float,
        out_path: str,
    ) -> None:
        """Slice waveform tensor and save to out_path as WAV."""
        import torchaudio

        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        clip = waveform[:, start_sample:end_sample]
        torchaudio.save(out_path, clip, sample_rate)

    def _make_segment_id(
        self, source_id: str, speaker: str, start: float, end: float
    ) -> str:
        """Return "{source_id}_{speaker}_{start:.2f}_{end:.2f}"."""
        return f"{source_id}_{speaker}_{start:.2f}_{end:.2f}"
