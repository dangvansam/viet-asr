import os
from datetime import datetime
from typing import Any, Dict

from loguru import logger

from .base import BaseReporter


class TextReporter(BaseReporter):
    def __init__(
        self,
        model_path: str = "",
        streaming: bool = False,
    ):
        self._model_path = model_path
        self._streaming = streaming

    def generate(
        self,
        results: Dict[str, Any],
        output_dir: str,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "report.txt")

        lines = self._build_header()

        for mode_name, result in results.items():
            lines.extend(self._build_mode_section(mode_name, result))

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Text report: {report_path}")

    def _build_header(self):
        return [
            "=" * 72,
            "SPEAKER DIARIZATION EVALUATION REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {self._model_path}",
            f"Mode: {'Streaming' if self._streaming else 'Non-streaming'}",
            "=" * 72,
            "",
        ]

    def _build_mode_section(self, mode_name: str, result: Dict[str, Any]):
        agg = result["aggregate"]
        settings = result["eval_settings"]
        lines = []

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

        if "latency" in result and result["latency"].get("aggregate"):
            lines.extend(self._build_latency_section(result["latency"]["aggregate"]))

        lines.extend(self._build_per_file_table(result["per_file"]))
        lines.append("")

        return lines

    def _build_latency_section(self, lat_agg: Dict[str, float]):
        return [
            "LATENCY METRICS (Streaming):",
            f"  Mean RTF:           {lat_agg.get('mean_rtf', 0):.4f}",
            f"  Mean Chunk Latency: {lat_agg.get('mean_chunk_ms', 0):.1f} ms",
            f"  P50 Chunk Latency:  {lat_agg.get('p50_chunk_ms', 0):.1f} ms",
            f"  P95 Chunk Latency:  {lat_agg.get('p95_chunk_ms', 0):.1f} ms",
            f"  Max Chunk Latency:  {lat_agg.get('max_chunk_ms', 0):.1f} ms",
            f"  Mean First Pred:    {lat_agg.get('mean_first_pred_ms', 0):.1f} ms",
            "",
        ]

    def _build_per_file_table(self, per_file):
        per_file = sorted(per_file, key=lambda x: x["DER"], reverse=True)
        lines = [
            "PER-FILE RESULTS (sorted by DER, worst first):",
            f"{'File ID':<25} {'DER%':>7} {'FA%':>7} {'Miss%':>7} "
            f"{'Conf%':>7} {'RefSpk':>6} {'HypSpk':>6} {'Dur(s)':>7}",
            "-" * 72,
        ]

        for pf in per_file:
            lines.append(
                f"{pf['file_id']:<25} {pf['DER'] * 100:7.2f} "
                f"{pf['FA'] * 100:7.2f} {pf['Miss'] * 100:7.2f} "
                f"{pf['Confusion'] * 100:7.2f} {pf['ref_spks']:6d} "
                f"{pf['hyp_spks']:6d} {pf['duration']:7.1f}"
            )

        return lines
