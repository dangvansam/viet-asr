import os
import json
from datetime import datetime

import numpy as np
from loguru import logger

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def generate_report(all_results, cfg):
    """Generate text report, charts, and Audacity labels.

    Args:
        all_results: dict from calculate_metrics(), keyed by eval mode name.
        cfg: EvalConfig with output_dir, generate_charts, generate_audacity_labels.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    _write_text_report(all_results, cfg)

    if cfg.generate_charts:
        if not HAS_MATPLOTLIB:
            logger.warning(
                "matplotlib not installed, skipping chart generation. "
                "Install with: uv add matplotlib"
            )
        else:
            _generate_charts(all_results, cfg)

    logger.info(f"Report generated in {cfg.output_dir}")


def _write_text_report(all_results, cfg):
    """Write a human-readable text report."""
    report_path = os.path.join(cfg.output_dir, "report.txt")
    lines = []

    lines.append("=" * 72)
    lines.append("SPEAKER DIARIZATION EVALUATION REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model: {cfg.diar_model_path}")
    lines.append(f"Mode: {'Streaming' if cfg.streaming else 'Non-streaming'}")
    lines.append("=" * 72)
    lines.append("")

    for mode_name, result in all_results.items():
        agg = result["aggregate"]
        settings = result["eval_settings"]

        lines.append("-" * 72)
        lines.append(
            f"Eval Mode: {mode_name.upper()} "
            f"(collar={settings['collar']}s, "
            f"ignore_overlap={settings['ignore_overlap']})"
        )
        lines.append("-" * 72)
        lines.append("")
        lines.append("AGGREGATE METRICS:")
        lines.append(f"  DER:              {agg['DER'] * 100:7.2f}%")
        lines.append(f"  False Alarm:      {agg['FA'] * 100:7.2f}%")
        lines.append(f"  Missed Detection: {agg['Miss'] * 100:7.2f}%")
        lines.append(f"  Confusion:        {agg['Confusion'] * 100:7.2f}%")
        lines.append(f"  Spk Count Acc:    {agg['spk_count_acc'] * 100:7.2f}%")
        lines.append(f"  Num Files:        {agg['num_files']}")
        lines.append("")

        # Latency stats (streaming only)
        if "latency" in result and result["latency"].get("aggregate"):
            lat_agg = result["latency"]["aggregate"]
            lines.append("LATENCY METRICS (Streaming):")
            lines.append(f"  Mean RTF:           {lat_agg.get('mean_rtf', 0):.4f}")
            lines.append(f"  Mean Chunk Latency: {lat_agg.get('mean_chunk_ms', 0):.1f} ms")
            lines.append(f"  P50 Chunk Latency:  {lat_agg.get('p50_chunk_ms', 0):.1f} ms")
            lines.append(f"  P95 Chunk Latency:  {lat_agg.get('p95_chunk_ms', 0):.1f} ms")
            lines.append(f"  Max Chunk Latency:  {lat_agg.get('max_chunk_ms', 0):.1f} ms")
            lines.append(f"  Mean First Pred:    {lat_agg.get('mean_first_pred_ms', 0):.1f} ms")
            lines.append("")

        # Per-file table (sorted by DER descending)
        per_file = sorted(
            result["per_file"], key=lambda x: x["DER"], reverse=True
        )
        lines.append("PER-FILE RESULTS (sorted by DER, worst first):")
        lines.append(
            f"{'File ID':<25} {'DER%':>7} {'FA%':>7} {'Miss%':>7} "
            f"{'Conf%':>7} {'RefSpk':>6} {'HypSpk':>6} {'Dur(s)':>7}"
        )
        lines.append("-" * 72)

        for pf in per_file:
            lines.append(
                f"{pf['file_id']:<25} {pf['DER'] * 100:7.2f} "
                f"{pf['FA'] * 100:7.2f} {pf['Miss'] * 100:7.2f} "
                f"{pf['Confusion'] * 100:7.2f} {pf['ref_spks']:6d} "
                f"{pf['hyp_spks']:6d} {pf['duration']:7.1f}"
            )

        lines.append("")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Text report: {report_path}")


def _generate_charts(all_results, cfg):
    """Generate matplotlib charts for the evaluation results."""
    charts_dir = os.path.join(cfg.output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # Use the first eval mode for charts (or "fair" if "all")
    if "fair" in all_results:
        result = all_results["fair"]
        mode_label = "fair"
    else:
        mode_label = next(iter(all_results))
        result = all_results[mode_label]

    per_file = result["per_file"]
    if not per_file:
        logger.warning("No per-file results, skipping charts")
        return

    ders = [pf["DER"] * 100 for pf in per_file]
    fas = [pf["FA"] * 100 for pf in per_file]
    misses = [pf["Miss"] * 100 for pf in per_file]
    confs = [pf["Confusion"] * 100 for pf in per_file]
    durations = [pf["duration"] for pf in per_file]
    ref_spks = [pf["ref_spks"] for pf in per_file]

    # 1. DER Distribution Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(ders, bins=30, edgecolor="black", alpha=0.7, color="#4C72B0")
    ax.axvline(np.mean(ders), color="red", linestyle="--", label=f"Mean: {np.mean(ders):.1f}%")
    ax.axvline(np.median(ders), color="orange", linestyle="--", label=f"Median: {np.median(ders):.1f}%")
    ax.set_xlabel("DER (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"DER Distribution [{mode_label}]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "der_distribution.png"), dpi=150)
    plt.close(fig)

    # 2. Error Breakdown Bar Chart (top 30 worst files)
    sorted_idx = np.argsort(ders)[::-1][:30]
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(sorted_idx))
    bar_width = 0.8
    ax.bar(x, [fas[i] for i in sorted_idx], bar_width, label="False Alarm", color="#E24A33")
    ax.bar(
        x,
        [misses[i] for i in sorted_idx],
        bar_width,
        bottom=[fas[i] for i in sorted_idx],
        label="Missed",
        color="#348ABD",
    )
    ax.bar(
        x,
        [confs[i] for i in sorted_idx],
        bar_width,
        bottom=[fas[i] + misses[i] for i in sorted_idx],
        label="Confusion",
        color="#FBC15E",
    )
    ax.set_xlabel("File (sorted by DER)")
    ax.set_ylabel("Error Rate (%)")
    ax.set_title(f"Error Breakdown - Top 30 Worst [{mode_label}]")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [per_file[i]["file_id"][-10:] for i in sorted_idx],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "error_breakdown.png"), dpi=150)
    plt.close(fig)

    # 3. DER by Number of Speakers (box plot)
    spk_groups = {}
    for pf in per_file:
        n = pf["ref_spks"]
        spk_groups.setdefault(n, []).append(pf["DER"] * 100)

    if spk_groups:
        fig, ax = plt.subplots(figsize=(8, 6))
        sorted_keys = sorted(spk_groups.keys())
        data = [spk_groups[k] for k in sorted_keys]
        bp = ax.boxplot(data, tick_labels=[str(k) for k in sorted_keys], patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#8EBA42")
        ax.set_xlabel("Number of Speakers")
        ax.set_ylabel("DER (%)")
        ax.set_title(f"DER by Speaker Count [{mode_label}]")
        fig.tight_layout()
        fig.savefig(os.path.join(charts_dir, "der_by_num_speakers.png"), dpi=150)
        plt.close(fig)

    # 4. DER vs Duration (scatter)
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        durations, ders, c=ref_spks, cmap="viridis", alpha=0.6, edgecolors="black", linewidth=0.5
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Num Speakers")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("DER (%)")
    ax.set_title(f"DER vs Audio Duration [{mode_label}]")
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "der_by_duration.png"), dpi=150)
    plt.close(fig)

    # 5 & 6. Latency charts (streaming only)
    if "latency" in result and result["latency"].get("per_file"):
        _generate_latency_charts(result, charts_dir, mode_label)

    logger.info(f"Charts saved to {charts_dir}")


def _generate_latency_charts(result, charts_dir, mode_label):
    """Generate streaming latency charts."""
    lat_data = result["latency"]

    # Collect all chunk latencies from the eval manifest
    # We need the raw chunk latencies which are in the per_file data
    per_file_lat = lat_data.get("per_file", [])
    if not per_file_lat:
        return

    # 5. Latency Distribution
    all_means = [pf["mean_chunk_ms"] for pf in per_file_lat]
    all_p95 = [pf["p95_chunk_ms"] for pf in per_file_lat]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(all_means, bins=25, edgecolor="black", alpha=0.7, color="#4C72B0")
    axes[0].axvline(np.mean(all_means), color="red", linestyle="--",
                    label=f"Mean: {np.mean(all_means):.1f}ms")
    axes[0].set_xlabel("Mean Chunk Latency (ms)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Per-File Mean Chunk Latency")
    axes[0].legend()

    rtfs = [pf["rtf"] for pf in per_file_lat]
    axes[1].hist(rtfs, bins=25, edgecolor="black", alpha=0.7, color="#E24A33")
    axes[1].axvline(np.mean(rtfs), color="blue", linestyle="--",
                    label=f"Mean RTF: {np.mean(rtfs):.3f}")
    axes[1].axvline(1.0, color="green", linestyle="-", linewidth=2,
                    label="Real-time (RTF=1.0)")
    axes[1].set_xlabel("Real-Time Factor (RTF)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("RTF Distribution")
    axes[1].legend()

    fig.suptitle(f"Streaming Latency [{mode_label}]")
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "latency_distribution.png"), dpi=150)
    plt.close(fig)

    # 6. First prediction latency
    first_preds = [pf["first_pred_ms"] for pf in per_file_lat]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(first_preds, bins=25, edgecolor="black", alpha=0.7, color="#8EBA42")
    ax.axvline(np.mean(first_preds), color="red", linestyle="--",
               label=f"Mean: {np.mean(first_preds):.1f}ms")
    ax.set_xlabel("Time to First Prediction (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"Time to First Prediction [{mode_label}]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "latency_first_pred.png"), dpi=150)
    plt.close(fig)


def rttm_to_audacity_labels(rttm_path, output_path):
    """Convert an RTTM file to Audacity label format.

    Audacity label format: start_seconds\\tend_seconds\\tlabel
    """
    lines = []
    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            end = start + duration
            lines.append(f"{start:.6f}\t{end:.6f}\t{speaker}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def generate_all_audacity_labels(eval_manifest, output_dir):
    """Generate Audacity label files for all hypothesis RTTMs."""
    labels_dir = os.path.join(output_dir, "audacity_labels")
    os.makedirs(labels_dir, exist_ok=True)

    for entry in eval_manifest:
        sample_id = entry["sample_id"]
        hyp_rttm = entry.get("hyp_rttm_filepath")
        if hyp_rttm and os.path.exists(hyp_rttm):
            out_path = os.path.join(labels_dir, f"{sample_id}_hyp.txt")
            rttm_to_audacity_labels(hyp_rttm, out_path)

    logger.info(f"Audacity labels saved to {labels_dir}")
