import os
import time
import math

import numpy as np
import soundfile as sf
import torch
from loguru import logger
from tqdm import tqdm

from nemo.collections.asr.models import SortformerEncLabelModel


def _parse_rttm_output(rttm_lines):
    """Parse RTTM lines returned by diarize() into a list of strings.

    The diarize() method may return different formats depending on version:
    - List[List[str]]: list of RTTM line lists per audio
    - Tuple[List[List[str]], List[Tensor]]: if include_tensor_outputs=True
    """
    if isinstance(rttm_lines, tuple):
        rttm_lines = rttm_lines[0]
    return rttm_lines


def _rttm_lines_to_audacity(rttm_lines):
    """Convert RTTM lines to Audacity label format."""
    labels = []
    for line in rttm_lines:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        start = float(parts[3])
        duration = float(parts[4])
        speaker = parts[7]
        end = start + duration
        labels.append(f"{start:.6f}\t{end:.6f}\t{speaker}")
    return labels


def _load_model(cfg):
    """Load and prepare the Sortformer diarization model."""
    logger.info(f"Loading diarization model from {cfg.diar_model_path}...")
    diar_model = SortformerEncLabelModel.restore_from(
        cfg.diar_model_path, map_location="cpu"
    )

    device = cfg.device
    if device == "cuda" and cfg.cuda_id >= 0:
        device = f"cuda:{cfg.cuda_id}"
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    diar_model = diar_model.to(device)
    diar_model.eval()
    logger.info(f"Model loaded on {device}")
    return diar_model, device


def run_diar_inference(cfg, eval_manifest):
    """Run diarization inference on eval set.

    Args:
        cfg: EvalConfig with diar_model_path, streaming, batch_size, output_dir.
        eval_manifest: List of dicts with audio_filepath, rttm_filepath, sample_id.

    Returns:
        Updated eval_manifest with hyp_rttm_filepath and latency stats added.
    """
    diar_model, device = _load_model(cfg)

    hyp_rttm_dir = os.path.join(cfg.output_dir, "hyp_rttm")
    hyp_labels_dir = os.path.join(cfg.output_dir, "audacity_labels")
    os.makedirs(hyp_rttm_dir, exist_ok=True)
    if cfg.generate_audacity_labels:
        os.makedirs(hyp_labels_dir, exist_ok=True)

    if cfg.streaming:
        return _run_streaming_inference(
            diar_model, device, cfg, eval_manifest, hyp_rttm_dir, hyp_labels_dir
        )
    else:
        return _run_offline_inference(
            diar_model, device, cfg, eval_manifest, hyp_rttm_dir, hyp_labels_dir
        )


def _run_offline_inference(
    diar_model, device, cfg, eval_manifest, hyp_rttm_dir, hyp_labels_dir
):
    """Non-streaming (offline) diarization using diarize() API."""
    results = []

    for entry in tqdm(eval_manifest, desc="Offline diarization"):
        audio_path = entry["audio_filepath"]
        sample_id = entry["sample_id"]

        t_start = time.perf_counter()
        outputs = diar_model.diarize(
            audio=[audio_path],
            batch_size=cfg.batch_size,
        )
        t_end = time.perf_counter()

        rttm_lines_list = _parse_rttm_output(outputs)
        rttm_lines = rttm_lines_list[0] if rttm_lines_list else []

        # Save hypothesis RTTM
        hyp_rttm_path = os.path.join(hyp_rttm_dir, f"{sample_id}.rttm")
        with open(hyp_rttm_path, "w") as f:
            for line in rttm_lines:
                f.write(line.strip() + "\n")

        # Save hypothesis Audacity labels
        if cfg.generate_audacity_labels:
            hyp_label_path = os.path.join(hyp_labels_dir, f"{sample_id}_hyp.txt")
            audacity_lines = _rttm_lines_to_audacity(rttm_lines)
            with open(hyp_label_path, "w") as f:
                f.write("\n".join(audacity_lines) + "\n")

        result = dict(entry)
        result["hyp_rttm_filepath"] = os.path.abspath(hyp_rttm_path)
        result["inference_time_s"] = round(t_end - t_start, 4)
        result["rtf"] = round(
            (t_end - t_start) / max(entry["duration"], 0.01), 4
        )
        results.append(result)

    return results


def _run_streaming_inference(
    diar_model, device, cfg, eval_manifest, hyp_rttm_dir, hyp_labels_dir
):
    """Streaming diarization with per-chunk latency measurement.

    Uses forward_streaming_step() to process audio chunk by chunk,
    measuring the latency of each chunk.
    """
    from omegaconf import OmegaConf
    from nemo.collections.asr.parts.utils.vad_utils import (
        ts_vad_post_processing,
        PostProcessingParams,
    )

    # Configure streaming parameters on the model
    OmegaConf.set_struct(diar_model.cfg, False)
    if not hasattr(diar_model.cfg, "stream_params"):
        diar_model.cfg.stream_params = OmegaConf.create({
            "window_length_s": 0.5,
            "shift_length_s": 0.05,
            "margin_frames": 10,
            "latency_s": 0.5,
        })

    # Get model parameters for chunking
    subsampling_factor = int(diar_model.cfg.encoder.subsampling_factor)
    n_spk = diar_model.sortformer_modules.n_spk
    chunk_len = diar_model.sortformer_modules.chunk_len

    results = []

    for entry in tqdm(eval_manifest, desc="Streaming diarization"):
        audio_path = entry["audio_filepath"]
        sample_id = entry["sample_id"]

        # Load and preprocess audio
        audio_samples, sr = sf.read(audio_path, dtype="float32")
        if len(audio_samples.shape) > 1:
            audio_samples = audio_samples.mean(axis=1)

        # Convert to tensor [batch=1, channel=1, samples]
        audio_tensor = torch.tensor(audio_samples, dtype=torch.float32).unsqueeze(0)
        audio_length = torch.tensor([audio_samples.shape[0]], dtype=torch.long)

        # Move to device
        audio_tensor = audio_tensor.to(device)
        audio_length = audio_length.to(device)

        # Run preprocessor to get features
        with torch.inference_mode():
            processed_signal, processed_signal_length = diar_model.preprocessor(
                input_signal=audio_tensor, length=audio_length
            )

        # processed_signal shape: (batch, channels, feat_len)
        # Run streaming inference with per-chunk timing
        chunk_latencies_ms = []

        with torch.inference_mode():
            streaming_state = diar_model.sortformer_modules.init_streaming_state(
                batch_size=1,
                async_streaming=diar_model.async_streaming if hasattr(diar_model, 'async_streaming') else False,
                device=device,
            )

            total_preds = torch.zeros((1, 0, n_spk), device=device)

            feat_len = processed_signal.shape[2]
            processed_signal_offset = torch.zeros((1,), dtype=torch.long, device=device)

            streaming_loader = diar_model.sortformer_modules.streaming_feat_loader(
                feat_seq=processed_signal,
                feat_seq_length=processed_signal_length,
                feat_seq_offset=processed_signal_offset,
            )

            first_pred_time = None
            t_file_start = time.perf_counter()

            for _, chunk_feat_seq_t, feat_lengths, left_offset, right_offset in streaming_loader:
                t_chunk_start = time.perf_counter()

                streaming_state, total_preds = diar_model.forward_streaming_step(
                    processed_signal=chunk_feat_seq_t,
                    processed_signal_length=feat_lengths,
                    streaming_state=streaming_state,
                    total_preds=total_preds,
                    left_offset=left_offset,
                    right_offset=right_offset,
                )

                t_chunk_end = time.perf_counter()
                chunk_ms = (t_chunk_end - t_chunk_start) * 1000
                chunk_latencies_ms.append(chunk_ms)

                if first_pred_time is None:
                    first_pred_time = t_chunk_end

            t_file_end = time.perf_counter()

        # Post-process predictions to RTTM
        rttm_lines = _postprocess_preds_to_rttm(
            total_preds, sample_id, diar_model, subsampling_factor
        )

        # Save hypothesis RTTM
        hyp_rttm_path = os.path.join(hyp_rttm_dir, f"{sample_id}.rttm")
        with open(hyp_rttm_path, "w") as f:
            for line in rttm_lines:
                f.write(line.strip() + "\n")

        # Save hypothesis Audacity labels
        if cfg.generate_audacity_labels:
            hyp_label_path = os.path.join(hyp_labels_dir, f"{sample_id}_hyp.txt")
            audacity_lines = _rttm_lines_to_audacity(rttm_lines)
            with open(hyp_label_path, "w") as f:
                f.write("\n".join(audacity_lines) + "\n")

        # Compute latency statistics
        total_time = t_file_end - t_file_start
        latencies = np.array(chunk_latencies_ms) if chunk_latencies_ms else np.array([0.0])

        result = dict(entry)
        result["hyp_rttm_filepath"] = os.path.abspath(hyp_rttm_path)
        result["inference_time_s"] = round(total_time, 4)
        result["rtf"] = round(total_time / max(entry["duration"], 0.01), 4)
        result["latency"] = {
            "num_chunks": len(chunk_latencies_ms),
            "first_pred_ms": round(
                (first_pred_time - t_file_start) * 1000, 2
            ) if first_pred_time else 0.0,
            "mean_chunk_ms": round(float(np.mean(latencies)), 2),
            "p50_chunk_ms": round(float(np.percentile(latencies, 50)), 2),
            "p95_chunk_ms": round(float(np.percentile(latencies, 95)), 2),
            "max_chunk_ms": round(float(np.max(latencies)), 2),
            "chunk_latencies_ms": [round(x, 2) for x in chunk_latencies_ms],
        }
        results.append(result)

    return results


def _postprocess_preds_to_rttm(total_preds, sample_id, diar_model, subsampling_factor):
    """Convert raw sigmoid predictions to RTTM lines.

    Applies threshold-based binarization and generates speaker timestamps.
    """
    from nemo.collections.asr.parts.utils.vad_utils import ts_vad_post_processing

    speaker_assign_mat = total_preds.squeeze(0)  # (T, n_spk)
    n_spk = speaker_assign_mat.shape[-1]
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
            # Fallback: simple threshold-based segmentation
            ts_mat = _simple_threshold_to_segments(
                spk_preds.cpu().numpy(), subsampling_factor, threshold=0.5
            )

        if len(ts_mat) == 0:
            continue

        for seg in ts_mat:
            if hasattr(seg, 'tolist'):
                seg = seg.tolist()
            start_t, end_t = seg[0], seg[1]
            duration = end_t - start_t
            if duration > 0.01:
                rttm_lines.append(
                    f"SPEAKER {sample_id} 1 {start_t:.2f} {duration:.2f} "
                    f"<NA> <NA> speaker_{spk_id} <NA> <NA>"
                )

    return rttm_lines


def _simple_threshold_to_segments(preds, subsampling_factor, threshold=0.5):
    """Fallback: convert frame-level predictions to time segments.

    Args:
        preds: numpy array of shape (T,) with sigmoid values
        subsampling_factor: encoder subsampling factor
        threshold: binarization threshold

    Returns:
        List of [start_time, end_time] segments.
    """
    frame_dur = subsampling_factor * 0.01  # 10ms per base frame
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
