"""
VADDiarizeStage: segment a long WAV into per-speaker turns and clip each to a
WAV file. Pluggable diarizer backend (config.diarize.backend):
  - "pyannote"  : pyannote/speaker-diarization-3.1 (speaker turns + overlap)
  - "sortformer": project NeMo Sortformer .nemo (native overlap, no HF gate)
  - "vad_sv"    : silero VAD segmentation only; speaker_verify assigns labels
Overlap turns are flagged (is_overlap) and kept; num_speakers is per source file.
"""

import gc
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

from ..base_stage import BaseStage
from ..checkpoint import PipelineCheckpoint
from ..config import DiarizeConfig, PipelineConfig, VADConfig
from ..parallel import parallel_map
from ..segment_utils import speech_span
from ..vad_backends import build_vad_backend


class VADDiarizeStage(BaseStage):
    """Segment audio into per-speaker turns via a pluggable diarizer backend."""

    name = "vad_diarize"

    def __init__(self) -> None:
        self._pipeline = None
        self._sortformer = None
        self._vad_backend = None

    def run(
        self,
        records: List[Dict],
        config: PipelineConfig,
        checkpoint: PipelineCheckpoint,
    ) -> List[Dict]:
        to_process, done = self._skip_processed(records, checkpoint)
        if not to_process:
            return done

        backend = config.diarize.backend
        self._load_diarizer(backend, config)
        if config.vad.enable_trim:
            self._load_vad_backend(config.vad)

        seg_dir = Path(config.output_dir) / "segments"
        seg_dir.mkdir(parents=True, exist_ok=True)

        all_segments: List[Dict] = []
        try:
            import soundfile as sf

            def process_file(record: Dict) -> List[Dict]:
                audio_path = record["audio_filepath"]
                try:
                    data, sample_rate = sf.read(audio_path, dtype="float32")
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    waveform = data[np.newaxis, :]  # [1, N] — numpy keeps the orchestrator torch-free

                    turns = self._diarize(backend, audio_path, waveform, sample_rate, config)
                    overlaps = self._mark_overlap(turns, config.diarize)
                    num_speakers = len({t[2] for t in turns})

                    file_segments = self._build_segments(
                        record, turns, overlaps, num_speakers,
                        waveform, sample_rate, seg_dir, config.vad,
                    )
                    if not file_segments:
                        logger.warning(f"No segments kept from {audio_path}")
                    logger.info(
                        f"Diarized {record['id']} ({backend}): "
                        f"{len(file_segments)} seg, {num_speakers} spk"
                    )
                    return file_segments
                except Exception as e:
                    logger.error(f"Diarization failed for {record['id']}: {e}")
                    return []

            workers = getattr(config, "concurrency", 1) or 1
            per_file = parallel_map(process_file, to_process, workers)
            for record, file_segments in zip(to_process, per_file):
                checkpoint.mark_processed(record["id"], self.name)
                all_segments.extend(file_segments)
            checkpoint.save_state()
        finally:
            self._free()

        return done + all_segments

    # ---- diarizer dispatch ------------------------------------------------

    def _diarize(
        self, backend: str, audio_path: str, waveform, sample_rate: int, config: PipelineConfig
    ) -> List[Tuple[float, float, str]]:
        if backend == "sortformer":
            return self._diarize_sortformer(audio_path)
        if backend == "vad_sv":
            return self._diarize_vad_sv(waveform, sample_rate, config.vad)
        return self._diarize_pyannote(waveform, sample_rate)

    def _diarize_pyannote(self, waveform, sample_rate: int) -> List[Tuple[float, float, str]]:
        import torch  # pyannote is a heavy local backend (torch present); vad_sv path never reaches here

        diar_out = self._pipeline(
            {"waveform": torch.from_numpy(np.asarray(waveform)), "sample_rate": sample_rate}
        )
        annotation = getattr(diar_out, "speaker_diarization", diar_out)
        return [
            (turn.start, turn.end, str(speaker))
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]

    def _diarize_sortformer(self, audio_path: str) -> List[Tuple[float, float, str]]:
        outputs = self._sortformer.diarize(audio=[audio_path])
        lines_list = outputs[0] if isinstance(outputs, tuple) else outputs
        lines = lines_list[0] if lines_list else []
        return [t for t in (self._parse_diar_line(ln) for ln in lines) if t]

    def _parse_diar_line(self, line: str):
        """Parse either full RTTM or NeMo's compact 'start end speaker' format."""
        parts = str(line).strip().split()
        if not parts:
            return None
        if parts[0] == "SPEAKER" and len(parts) >= 8:
            start = float(parts[3])
            return (start, start + float(parts[4]), str(parts[7]))
        if len(parts) >= 3:
            try:
                return (float(parts[0]), float(parts[1]), str(parts[2]))
            except ValueError:
                return None
        return None

    def _diarize_vad_sv(
        self, waveform, sample_rate: int, vad_config: VADConfig
    ) -> List[Tuple[float, float, str]]:
        """VAD-only segmentation; placeholder speaker labels (speaker_verify relabels)."""
        if self._vad_backend is None:
            self._load_vad_backend(vad_config)
        audio = waveform.mean(axis=0)
        result = self._vad_backend.detect(audio, sample_rate)
        return [
            (seg.start, seg.end, f"SEG_{i:04d}")
            for i, seg in enumerate(result.segments)
        ]

    def _mark_overlap(self, turns, diar_cfg: DiarizeConfig) -> List[bool]:
        """A turn is overlap if it intersects another speaker's turn >= overlap_min_s."""
        n = len(turns)
        flags = [False] * n
        if not diar_cfg.detect_overlap:
            return flags
        for i in range(n):
            si, ei, spi = turns[i]
            for j in range(n):
                if i == j:
                    continue
                sj, ej, spj = turns[j]
                if spj == spi:
                    continue
                inter = min(ei, ej) - max(si, sj)
                if inter >= diar_cfg.overlap_min_s:
                    flags[i] = True
                    break
        return flags

    # ---- segment building -------------------------------------------------

    def _build_segments(
        self, record, turns, overlaps, num_speakers,
        waveform, sample_rate, seg_dir, vad_config,
    ) -> List[Dict]:
        file_segments: List[Dict] = []
        file_vad = record.get("vad_segments")
        for (start, end, speaker), is_overlap in zip(turns, overlaps):
            trim_meta = None
            if vad_config.enable_trim and not is_overlap:
                start, end, trim_meta = self._trim_turn(
                    file_vad, waveform, sample_rate, start, end, vad_config
                )
            dur = end - start
            if dur < vad_config.min_duration or dur > vad_config.max_duration:
                continue

            seg_id = self._make_segment_id(record["id"], speaker, start, end)
            out_path = str(seg_dir / f"{seg_id}.wav")
            self._clip_segment(waveform, sample_rate, start, end, out_path)

            seg_record = dict(record)
            seg_record.update({
                "id": seg_id,
                "audio_filepath": out_path,
                "source_audio": record["audio_filepath"],
                "start": start,
                "end": end,
                "speaker_id": speaker,
                "duration": dur,
                "num_speakers": num_speakers,
            })
            seg_extra = dict(record.get("extra") or {})
            seg_extra["is_overlap"] = bool(is_overlap)
            if trim_meta is not None:
                seg_extra["trim"] = trim_meta
                seg_record["vad_segments"] = trim_meta["vad_segments"]
            seg_record["extra"] = seg_extra
            file_segments.append(seg_record)
        return file_segments

    # ---- loaders ----------------------------------------------------------

    def _load_diarizer(self, backend: str, config: PipelineConfig) -> None:
        if backend == "pyannote":
            if not config.vad.hf_token:
                logger.warning("HF_TOKEN not set — pyannote uses cached login token")
            self._pipeline = self._load_pyannote(config.vad, config.diarize.device)
        elif backend == "sortformer":
            self._sortformer = self._load_sortformer(config.diarize)
        elif backend == "vad_sv":
            self._load_vad_backend(config.vad)
        else:
            raise ValueError(f"Unknown diarize backend '{backend}'")

    def _load_sortformer(self, diar_cfg: DiarizeConfig):
        from nemo.collections.asr.models import SortformerEncLabelModel

        logger.info(f"Loading Sortformer diarizer: {diar_cfg.model_path}")
        model = SortformerEncLabelModel.restore_from(
            restore_path=diar_cfg.model_path,
            map_location=diar_cfg.device,
        )
        model.eval()
        return model

    def _load_pyannote(self, vad_config: VADConfig, device: str):
        import torch
        from pyannote.audio import Pipeline

        kwargs = {}
        if vad_config.hf_token:
            kwargs["use_auth_token"] = vad_config.hf_token
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", **kwargs)
        pipeline.to(torch.device(device))
        return pipeline

    def _load_vad_backend(self, vad_config: VADConfig) -> None:
        if self._vad_backend is not None:
            return
        kwargs = dict(vad_config.backend_kwargs)
        if vad_config.backend in ("pyannote_seg", "consensus") and vad_config.hf_token:
            kwargs.setdefault("hf_token", vad_config.hf_token)
        self._vad_backend = build_vad_backend(vad_config.backend, **kwargs)
        self._vad_backend.load(device=vad_config.device)

    def _free(self) -> None:
        self._pipeline = None
        self._sortformer = None
        if self._vad_backend is not None:
            self._vad_backend.unload()
            self._vad_backend = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- helpers (unchanged) ---------------------------------------------

    def _local_speech(self, file_vad, start, end):
        if not file_vad:
            return None
        local = []
        for seg in file_vad:
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", 0.0))
            lo = max(s, start)
            hi = min(e, end)
            if hi > lo:
                local.append((lo - start, hi - start))
        return local

    def _trim_turn(self, file_vad, waveform, sample_rate, start, end, vad_config):
        local = self._local_speech(file_vad, start, end)
        if local is None:
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            clip = waveform[:, start_sample:end_sample].mean(axis=0)
            try:
                result = self._vad_backend.detect(clip, sample_rate)
            except Exception as exc:
                logger.warning(f"VAD trim failed, keeping untrimmed turn: {exc}")
                return start, end, None
            local = [(seg.start, seg.end) for seg in result.segments]
        span = speech_span(local, pad_s=vad_config.trim_pad_s, lo=0.0, hi=end - start)
        if span is None:
            return start, end, None
        new_start = start + span[0]
        new_end = start + span[1]
        meta = {
            "original_start": start,
            "original_end": end,
            "vad_segments": [{"start": new_start, "end": new_end}],
        }
        return new_start, new_end, meta

    def _clip_segment(self, waveform, sample_rate, start, end, out_path) -> None:
        import soundfile as sf

        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        clip = waveform[:, start_sample:end_sample].squeeze(0)
        sf.write(out_path, clip, sample_rate)

    def _make_segment_id(self, source_id: str, speaker: str, start: float, end: float) -> str:
        return f"{source_id}_{speaker}_{start:.2f}_{end:.2f}"
