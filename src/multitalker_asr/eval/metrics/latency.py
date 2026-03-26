from typing import Any, Dict, List

import numpy as np

from .base import BaseMetric


class LatencyMetric(BaseMetric):
    def compute(
        self,
        reference: Any,
        hypothesis: Any,
    ) -> Dict[str, float]:
        return {}

    def aggregate(
        self,
        per_file_results: List[Dict[str, float]],
    ) -> Dict[str, float]:
        return self.aggregate_latency(per_file_results)

    def aggregate_latency(
        self,
        eval_manifest: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        file_latencies = []
        all_chunk_latencies = []

        for entry in eval_manifest:
            lat = entry.get("latency")
            if lat is None:
                continue
            file_latencies.append(
                {
                    "file_id": entry["sample_id"],
                    "rtf": entry.get("rtf", 0.0),
                    "mean_chunk_ms": lat["mean_chunk_ms"],
                    "p95_chunk_ms": lat["p95_chunk_ms"],
                    "first_pred_ms": lat["first_pred_ms"],
                }
            )
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
                    float(
                        np.mean([f["first_pred_ms"] for f in file_latencies])
                    ),
                    2,
                ),
            },
            "per_file": file_latencies,
        }
