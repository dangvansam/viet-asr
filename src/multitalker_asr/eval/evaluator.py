import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from loguru import logger
from nemo.collections.asr.models import SortformerEncLabelModel
from omegaconf import OmegaConf
from tqdm import tqdm

from ..configs import EvalConfig
from ..utils import AudacityConverter, DeviceManager
from .base import BaseEvaluator


class DiarizationEvaluator(BaseEvaluator):
    def __init__(self, cfg: EvalConfig):
        self._cfg = cfg
        self._model = None
        self._device = None

    def evaluate(self, eval_manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self._load_model()

        hyp_rttm_dir = os.path.join(self._cfg.output_dir, "hyp_rttm")
        hyp_labels_dir = os.path.join(self._cfg.output_dir, "audacity_labels")
        os.makedirs(hyp_rttm_dir, exist_ok=True)
        if self._cfg.generate_audacity_labels:
            os.makedirs(hyp_labels_dir, exist_ok=True)

        if self._cfg.streaming:
            return self._run_streaming(eval_manifest, hyp_rttm_dir, hyp_labels_dir)
        return self._run_offline(eval_manifest, hyp_rttm_dir, hyp_labels_dir)

    def _load_model(self) -> None:
        logger.info(f"Loading diarization model from {self._cfg.diar_model_path}...")
        self._model = SortformerEncLabelModel.restore_from(
            self._cfg.diar_model_path, map_location="cpu"
        )

        device_str = self._cfg.device
        if device_str == "cuda" and self._cfg.cuda_id >= 0:
            device_str = f"cuda:{self._cfg.cuda_id}"
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device_str = "cpu"

        self._device = device_str
        self._model = self._model.to(self._device)
        self._model.eval()
        logger.info(f"Model loaded on {self._device}")

    def _run_offline(self, eval_manifest, hyp_rttm_dir, hyp_labels_dir):
        results = []

        for entry in tqdm(eval_manifest, desc="Offline diarization"):
            audio_path = entry["audio_filepath"]
            sample_id = entry["sample_id"]

            t_start = time.perf_counter()
            outputs = self._model.diarize(audio=[audio_path], batch_size=self._cfg.batch_size)
            t_end = time.perf_counter()

            rttm_lines_list = self._parse_rttm_output(outputs)
            rttm_lines = rttm_lines_list[0] if rttm_lines_list else []

            hyp_rttm_path = os.path.join(hyp_rttm_dir, f"{sample_id}.rttm")
            self._write_rttm(rttm_lines, hyp_rttm_path)

            if self._cfg.generate_audacity_labels:
                hyp_label_path = os.path.join(hyp_labels_dir, f"{sample_id}_hyp.txt")
                AudacityConverter.from_rttm_lines(rttm_lines, hyp_label_path)

            result = dict(entry)
            result["hyp_rttm_filepath"] = os.path.abspath(hyp_rttm_path)
            result["inference_time_s"] = round(t_end - t_start, 4)
            result["rtf"] = round((t_end - t_start) / max(entry["duration"], 0.01), 4)
            results.append(result)

        return results

    def _run_streaming(self, eval_manifest, hyp_rttm_dir, hyp_labels_dir):
        OmegaConf.set_struct(self._model.cfg, False)
        if not hasattr(self._model.cfg, "stream_params"):
            self._model.cfg.stream_params = OmegaConf.create({
                "window_length_s": 0.5,
                "shift_length_s": 0.05,
                "margin_frames": 10,
                "latency_s": 0.5,
            })

        subsampling_factor = int(self._model.cfg.encoder.subsampling_factor)
        n_spk = self._model.sortformer_modules.n_spk

        results = []

        for entry in tqdm(eval_manifest, desc="Streaming diarization"):
            audio_path = entry["audio_filepath"]
            sample_id = entry["sample_id"]

            audio_samples, sr = sf.read(audio_path, dtype="float32")
            if len(audio_samples.shape) > 1:
                audio_samples = audio_samples.mean(axis=1)

            audio_tensor = torch.tensor(audio_samples, dtype=torch.float32).unsqueeze(0).to(self._device)
            audio_length = torch.tensor([audio_samples.shape[0]], dtype=torch.long).to(self._device)

            with torch.inference_mode():
                processed_signal, processed_signal_length = self._model.preprocessor(
                    input_signal=audio_tensor, length=audio_length
                )

            chunk_latencies_ms = []

            with torch.inference_mode():
                streaming_state = self._model.sortformer_modules.init_streaming_state(
                    batch_size=1,
                    async_streaming=getattr(self._model, "async_streaming", False),
                    device=self._device,
                )

                total_preds = torch.zeros((1, 0, n_spk), device=self._device)
                processed_signal_offset = torch.zeros((1,), dtype=torch.long, device=self._device)

                streaming_loader = self._model.sortformer_modules.streaming_feat_loader(
                    feat_seq=processed_signal,
                    feat_seq_length=processed_signal_length,
                    feat_seq_offset=processed_signal_offset,
                )

                first_pred_time = None
                t_file_start = time.perf_counter()

                for _, chunk_feat_seq_t, feat_lengths, left_offset, right_offset in streaming_loader:
                    t_chunk_start = time.perf_counter()

                    streaming_state, total_preds = self._model.forward_streaming_step(
                        processed_signal=chunk_feat_seq_t,
                        processed_signal_length=feat_lengths,
                        streaming_state=streaming_state,
                        total_preds=total_preds,
                        left_offset=left_offset,
                        right_offset=right_offset,
                    )

                    t_chunk_end = time.perf_counter()
                    chunk_latencies_ms.append((t_chunk_end - t_chunk_start) * 1000)

                    if first_pred_time is None:
                        first_pred_time = t_chunk_end

                t_file_end = time.perf_counter()

            rttm_lines = self._postprocess_predictions(
                total_preds, sample_id, subsampling_factor, n_spk
            )

            hyp_rttm_path = os.path.join(hyp_rttm_dir, f"{sample_id}.rttm")
            self._write_rttm(rttm_lines, hyp_rttm_path)

            if self._cfg.generate_audacity_labels:
                hyp_label_path = os.path.join(hyp_labels_dir, f"{sample_id}_hyp.txt")
                AudacityConverter.from_rttm_lines(rttm_lines, hyp_label_path)

            total_time = t_file_end - t_file_start
            latencies = np.array(chunk_latencies_ms) if chunk_latencies_ms else np.array([0.0])

            result = dict(entry)
            result["hyp_rttm_filepath"] = os.path.abspath(hyp_rttm_path)
            result["inference_time_s"] = round(total_time, 4)
            result["rtf"] = round(total_time / max(entry["duration"], 0.01), 4)
            result["latency"] = {
                "num_chunks": len(chunk_latencies_ms),
                "first_pred_ms": round((first_pred_time - t_file_start) * 1000, 2) if first_pred_time else 0.0,
                "mean_chunk_ms": round(float(np.mean(latencies)), 2),
                "p50_chunk_ms": round(float(np.percentile(latencies, 50)), 2),
                "p95_chunk_ms": round(float(np.percentile(latencies, 95)), 2),
                "max_chunk_ms": round(float(np.max(latencies)), 2),
                "chunk_latencies_ms": [round(x, 2) for x in chunk_latencies_ms],
            }
            results.append(result)

        return results

    def _postprocess_predictions(self, total_preds, sample_id, subsampling_factor, n_spk):
        from nemo.collections.asr.parts.utils.vad_utils import ts_vad_post_processing

        speaker_assign_mat = total_preds.squeeze(0)
        rttm_lines = []

        for spk_id in range(n_spk):
            spk_preds = speaker_assign_mat[:, spk_id]

            try:
                ts_mat = ts_vad_post_processing(
                    spk_preds,
                    cfg_vad_params=None,
                    unit_10ms_frame_count=subsampling_factor,
                    bypass_postprocessing=False,
                )
            except Exception:
                ts_mat = self._simple_threshold_to_segments(
                    spk_preds.cpu().numpy(), subsampling_factor
                )

            if len(ts_mat) == 0:
                continue

            for seg in ts_mat:
                if hasattr(seg, "tolist"):
                    seg = seg.tolist()
                start_t, end_t = seg[0], seg[1]
                duration = end_t - start_t
                if duration > 0.01:
                    rttm_lines.append(
                        f"SPEAKER {sample_id} 1 {start_t:.2f} {duration:.2f} "
                        f"<NA> <NA> speaker_{spk_id} <NA> <NA>"
                    )

        return rttm_lines

    def _simple_threshold_to_segments(self, preds, subsampling_factor, threshold=0.5):
        frame_dur = subsampling_factor * 0.01
        binary = (preds >= threshold).astype(int)
        segments = []
        in_segment = False
        start = 0.0

        for i, val in enumerate(binary):
            t = i * frame_dur
            if val == 1 and not in_segment:
                start = t
                in_segment = True
            elif val == 0 and in_segment:
                segments.append([start, t])
                in_segment = False

        if in_segment:
            segments.append([start, len(binary) * frame_dur])

        return segments

    @staticmethod
    def _parse_rttm_output(rttm_lines):
        if isinstance(rttm_lines, tuple):
            rttm_lines = rttm_lines[0]
        return rttm_lines

    @staticmethod
    def _write_rttm(rttm_lines, output_path):
        with open(output_path, "w") as f:
            for line in rttm_lines:
                f.write(line.strip() + "\n")
