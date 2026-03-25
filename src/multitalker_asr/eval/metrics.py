import os
import json

import numpy as np
from loguru import logger
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate


def rttm_to_pyannote_annotation(rttm_path, uri=""):
    """Parse an RTTM file into a pyannote Annotation object.

    RTTM format: SPEAKER <file_id> <channel> <start> <duration> <NA> <NA> <speaker> <NA> <NA>
    """
    annotation = Annotation(uri=uri)
    if not os.path.exists(rttm_path):
        logger.warning(f"RTTM file not found: {rttm_path}")
        return annotation

    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            if duration > 0:
                annotation[Segment(start, start + duration)] = speaker

    return annotation


def compute_per_file_der(ref_annotation, hyp_annotation, collar, ignore_overlap):
    """Compute DER breakdown for a single file.

    Args:
        ref_annotation: pyannote Annotation (ground truth)
        hyp_annotation: pyannote Annotation (prediction)
        collar: collar in seconds (pyannote uses full collar, not half)
        ignore_overlap: whether to skip overlap regions

    Returns:
        dict with DER, FA, Miss, Confusion, ref_spks, hyp_spks
    """
    metric = DiarizationErrorRate(collar=collar, skip_overlap=ignore_overlap)
    detail = metric(ref_annotation, hyp_annotation, detailed=True)

    total = detail["total"]
    if total == 0:
        return {
            "DER": 0.0,
            "FA": 0.0,
            "Miss": 0.0,
            "Confusion": 0.0,
            "ref_spks": len(ref_annotation.labels()),
            "hyp_spks": len(hyp_annotation.labels()),
        }

    return {
        "DER": round(detail["false alarm"] / total + detail["missed detection"] / total + detail["confusion"] / total, 4),
        "FA": round(detail["false alarm"] / total, 4),
        "Miss": round(detail["missed detection"] / total, 4),
        "Confusion": round(detail["confusion"] / total, 4),
        "ref_spks": len(ref_annotation.labels()),
        "hyp_spks": len(hyp_annotation.labels()),
    }


def _get_eval_settings(eval_mode):
    """Get (collar, ignore_overlap) pairs for the given eval mode.

    Note: pyannote uses collar as full width, while NeMo's score_labels()
    doubles the collar. We follow pyannote convention here (collar = 2 * half_collar).
    """
    if eval_mode == "full":
        return [("full", 0.0, False)]
    elif eval_mode == "fair":
        return [("fair", 0.5, False)]  # 0.5s = 2 * 0.25s collar
    elif eval_mode == "forgiving":
        return [("forgiving", 0.5, True)]
    elif eval_mode == "all":
        return [
            ("full", 0.0, False),
            ("fair", 0.5, False),
            ("forgiving", 0.5, True),
        ]
    else:
        raise ValueError(f"Unknown eval_mode: {eval_mode}")


def calculate_metrics(eval_manifest, cfg):
    """Calculate DER metrics across the eval set.

    Args:
        eval_manifest: List of dicts with rttm_filepath, hyp_rttm_filepath,
                       sample_id, duration, num_speakers.
        cfg: EvalConfig with collar, ignore_overlap, eval_mode, streaming.

    Returns:
        dict with results per eval setting, each containing aggregate and per_file metrics.
    """
    eval_settings = _get_eval_settings(cfg.eval_mode)
    all_results = {}

    for mode_name, collar, ignore_overlap in eval_settings:
        logger.info(
            f"Computing DER [{mode_name}]: collar={collar}s, "
            f"ignore_overlap={ignore_overlap}"
        )

        per_file = []
        correct_spk_count = 0

        # Cumulative metric for aggregate DER
        cumulative_metric = DiarizationErrorRate(
            collar=collar, skip_overlap=ignore_overlap
        )

        for entry in eval_manifest:
            sample_id = entry["sample_id"]
            ref_ann = rttm_to_pyannote_annotation(
                entry["rttm_filepath"], uri=sample_id
            )
            hyp_ann = rttm_to_pyannote_annotation(
                entry["hyp_rttm_filepath"], uri=sample_id
            )

            # Per-file detailed metrics
            file_metrics = compute_per_file_der(
                ref_ann, hyp_ann, collar, ignore_overlap
            )
            file_metrics["file_id"] = sample_id
            file_metrics["duration"] = entry.get("duration", 0.0)

            if file_metrics["ref_spks"] == file_metrics["hyp_spks"]:
                correct_spk_count += 1

            per_file.append(file_metrics)

            # Feed into cumulative metric
            cumulative_metric(ref_ann, hyp_ann, detailed=True)

        # Aggregate
        total = cumulative_metric["total"]
        if total > 0:
            aggregate = {
                "DER": round(abs(cumulative_metric), 4),
                "FA": round(cumulative_metric["false alarm"] / total, 4),
                "Miss": round(cumulative_metric["missed detection"] / total, 4),
                "Confusion": round(cumulative_metric["confusion"] / total, 4),
                "spk_count_acc": round(
                    correct_spk_count / max(len(eval_manifest), 1), 4
                ),
                "num_files": len(eval_manifest),
            }
        else:
            aggregate = {
                "DER": 0.0, "FA": 0.0, "Miss": 0.0, "Confusion": 0.0,
                "spk_count_acc": 0.0, "num_files": len(eval_manifest),
            }

        eval_result = {
            "aggregate": aggregate,
            "per_file": per_file,
            "eval_settings": {
                "mode": mode_name,
                "collar": collar,
                "ignore_overlap": ignore_overlap,
                "streaming": cfg.streaming,
            },
        }

        # Add latency stats if streaming
        if cfg.streaming:
            eval_result["latency"] = _aggregate_latency(eval_manifest)

        all_results[mode_name] = eval_result

        logger.info(
            f"[{mode_name}] DER={aggregate['DER']:.4f} "
            f"(FA={aggregate['FA']:.4f}, Miss={aggregate['Miss']:.4f}, "
            f"Conf={aggregate['Confusion']:.4f}) "
            f"SpkAcc={aggregate['spk_count_acc']:.4f}"
        )

    return all_results


def _aggregate_latency(eval_manifest):
    """Aggregate latency statistics across all files."""
    file_latencies = []
    all_chunk_latencies = []

    for entry in eval_manifest:
        lat = entry.get("latency")
        if lat is None:
            continue
        file_latencies.append({
            "file_id": entry["sample_id"],
            "rtf": entry.get("rtf", 0.0),
            "mean_chunk_ms": lat["mean_chunk_ms"],
            "p95_chunk_ms": lat["p95_chunk_ms"],
            "first_pred_ms": lat["first_pred_ms"],
        })
        all_chunk_latencies.extend(lat["chunk_latencies_ms"])

    if not all_chunk_latencies:
        return {"aggregate": {}, "per_file": file_latencies}

    chunks = np.array(all_chunk_latencies)
    rtfs = [f["rtf"] for f in file_latencies]

    return {
        "aggregate": {
            "mean_rtf": round(float(np.mean(rtfs)), 4),
            "mean_chunk_ms": round(float(np.mean(chunks)), 2),
            "p50_chunk_ms": round(float(np.percentile(chunks, 50)), 2),
            "p95_chunk_ms": round(float(np.percentile(chunks, 95)), 2),
            "max_chunk_ms": round(float(np.max(chunks)), 2),
            "mean_first_pred_ms": round(
                float(np.mean([f["first_pred_ms"] for f in file_latencies])), 2
            ),
        },
        "per_file": file_latencies,
    }


def save_metrics(all_results, output_dir):
    """Save metrics to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Metrics saved to {out_path}")
