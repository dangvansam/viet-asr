import os
from typing import Any, Dict, List

import numpy as np
from loguru import logger

from .base import BaseReporter

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ChartReporter(BaseReporter):
    def generate(
        self,
        results: Dict[str, Any],
        output_dir: str,
    ) -> None:
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not installed, skipping chart generation.")
            return

        charts_dir = os.path.join(output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)

        mode_label, result = self._select_mode(results)
        per_file = result["per_file"]

        if not per_file:
            logger.warning("No per-file results, skipping charts")
            return

        self._generate_der_distribution(per_file, charts_dir, mode_label)
        self._generate_error_breakdown(per_file, charts_dir, mode_label)
        self._generate_der_by_speakers(per_file, charts_dir, mode_label)
        self._generate_der_by_duration(per_file, charts_dir, mode_label)

        if "latency" in result and result["latency"].get("per_file"):
            self._generate_latency_charts(result["latency"], charts_dir, mode_label)

        logger.info(f"Charts saved to {charts_dir}")

    def _select_mode(self, results: Dict[str, Any]):
        if "fair" in results:
            return "fair", results["fair"]
        mode_label = next(iter(results))
        return mode_label, results[mode_label]

    def _generate_der_distribution(self, per_file, charts_dir, mode_label):
        ders = [pf["DER"] * 100 for pf in per_file]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(ders, bins=30, edgecolor="black", alpha=0.7, color="#4C72B0")
        ax.axvline(
            np.mean(ders), color="red", linestyle="--",
            label=f"Mean: {np.mean(ders):.1f}%",
        )
        ax.axvline(
            np.median(ders), color="orange", linestyle="--",
            label=f"Median: {np.median(ders):.1f}%",
        )
        ax.set_xlabel("DER (%)")
        ax.set_ylabel("Count")
        ax.set_title(f"DER Distribution [{mode_label}]")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(charts_dir, "der_distribution.png"), dpi=150)
        plt.close(fig)

    def _generate_error_breakdown(self, per_file, charts_dir, mode_label):
        ders = [pf["DER"] * 100 for pf in per_file]
        fas = [pf["FA"] * 100 for pf in per_file]
        misses = [pf["Miss"] * 100 for pf in per_file]
        confs = [pf["Confusion"] * 100 for pf in per_file]

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
            rotation=45, ha="right", fontsize=7,
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(charts_dir, "error_breakdown.png"), dpi=150)
        plt.close(fig)

    def _generate_der_by_speakers(self, per_file, charts_dir, mode_label):
        spk_groups = {}
        for pf in per_file:
            n = pf["ref_spks"]
            spk_groups.setdefault(n, []).append(pf["DER"] * 100)

        if not spk_groups:
            return

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

    def _generate_der_by_duration(self, per_file, charts_dir, mode_label):
        ders = [pf["DER"] * 100 for pf in per_file]
        durations = [pf["duration"] for pf in per_file]
        ref_spks = [pf["ref_spks"] for pf in per_file]

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            durations, ders, c=ref_spks, cmap="viridis",
            alpha=0.6, edgecolors="black", linewidth=0.5,
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Num Speakers")
        ax.set_xlabel("Duration (s)")
        ax.set_ylabel("DER (%)")
        ax.set_title(f"DER vs Audio Duration [{mode_label}]")
        fig.tight_layout()
        fig.savefig(os.path.join(charts_dir, "der_by_duration.png"), dpi=150)
        plt.close(fig)

    def _generate_latency_charts(self, lat_data, charts_dir, mode_label):
        per_file_lat = lat_data.get("per_file", [])
        if not per_file_lat:
            return

        all_means = [pf["mean_chunk_ms"] for pf in per_file_lat]
        rtfs = [pf["rtf"] for pf in per_file_lat]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(all_means, bins=25, edgecolor="black", alpha=0.7, color="#4C72B0")
        axes[0].axvline(
            np.mean(all_means), color="red", linestyle="--",
            label=f"Mean: {np.mean(all_means):.1f}ms",
        )
        axes[0].set_xlabel("Mean Chunk Latency (ms)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Per-File Mean Chunk Latency")
        axes[0].legend()

        axes[1].hist(rtfs, bins=25, edgecolor="black", alpha=0.7, color="#E24A33")
        axes[1].axvline(
            np.mean(rtfs), color="blue", linestyle="--",
            label=f"Mean RTF: {np.mean(rtfs):.3f}",
        )
        axes[1].axvline(1.0, color="green", linestyle="-", linewidth=2, label="Real-time (RTF=1.0)")
        axes[1].set_xlabel("Real-Time Factor (RTF)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("RTF Distribution")
        axes[1].legend()

        fig.suptitle(f"Streaming Latency [{mode_label}]")
        fig.tight_layout()
        fig.savefig(os.path.join(charts_dir, "latency_distribution.png"), dpi=150)
        plt.close(fig)

        first_preds = [pf["first_pred_ms"] for pf in per_file_lat]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(first_preds, bins=25, edgecolor="black", alpha=0.7, color="#8EBA42")
        ax.axvline(
            np.mean(first_preds), color="red", linestyle="--",
            label=f"Mean: {np.mean(first_preds):.1f}ms",
        )
        ax.set_xlabel("Time to First Prediction (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"Time to First Prediction [{mode_label}]")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(charts_dir, "latency_first_pred.png"), dpi=150)
        plt.close(fig)
