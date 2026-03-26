from typing import Any, Dict, List, Tuple

from loguru import logger
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

from ...utils.formats.pyannote import PyannoteConverter
from .base import BaseMetric

EVAL_MODES = {
    "full": [("full", 0.0, False)],
    "fair": [("fair", 0.5, False)],
    "forgiving": [("forgiving", 0.5, True)],
    "all": [
        ("full", 0.0, False),
        ("fair", 0.5, False),
        ("forgiving", 0.5, True),
    ],
}


class DERMetric(BaseMetric):
    def __init__(self, eval_mode: str = "all"):
        self._eval_mode = eval_mode
        self._settings = EVAL_MODES.get(eval_mode)
        if self._settings is None:
            raise ValueError(f"Unknown eval_mode: {eval_mode}")

    def compute(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        collar: float = 0.0,
        ignore_overlap: bool = False,
    ) -> Dict[str, float]:
        metric = DiarizationErrorRate(collar=collar, skip_overlap=ignore_overlap)
        detail = metric(reference, hypothesis, detailed=True)

        total = detail["total"]
        if total == 0:
            return {
                "DER": 0.0,
                "FA": 0.0,
                "Miss": 0.0,
                "Confusion": 0.0,
                "ref_spks": len(reference.labels()),
                "hyp_spks": len(hypothesis.labels()),
            }

        return {
            "DER": round(
                detail["false alarm"] / total
                + detail["missed detection"] / total
                + detail["confusion"] / total,
                4,
            ),
            "FA": round(detail["false alarm"] / total, 4),
            "Miss": round(detail["missed detection"] / total, 4),
            "Confusion": round(detail["confusion"] / total, 4),
            "ref_spks": len(reference.labels()),
            "hyp_spks": len(hypothesis.labels()),
        }

    def aggregate(
        self,
        per_file_results: List[Dict[str, float]],
    ) -> Dict[str, float]:
        if not per_file_results:
            return {"DER": 0.0, "FA": 0.0, "Miss": 0.0, "Confusion": 0.0}

        import numpy as np

        ders = [r["DER"] for r in per_file_results]
        return {
            "DER_mean": round(float(np.mean(ders)), 4),
            "DER_median": round(float(np.median(ders)), 4),
            "DER_std": round(float(np.std(ders)), 4),
        }

    def compute_all_modes(
        self,
        eval_manifest: List[Dict[str, Any]],
        streaming: bool = False,
    ) -> Dict[str, Any]:
        all_results = {}

        for mode_name, collar, ignore_overlap in self._settings:
            logger.info(
                f"Computing DER [{mode_name}]: collar={collar}s, "
                f"ignore_overlap={ignore_overlap}"
            )

            per_file = []
            correct_spk_count = 0

            cumulative_metric = DiarizationErrorRate(
                collar=collar, skip_overlap=ignore_overlap
            )

            for entry in eval_manifest:
                sample_id = entry["sample_id"]
                ref_ann = PyannoteConverter.rttm_to_annotation(
                    entry["rttm_filepath"], uri=sample_id
                )
                hyp_ann = PyannoteConverter.rttm_to_annotation(
                    entry["hyp_rttm_filepath"], uri=sample_id
                )

                file_metrics = self.compute(ref_ann, hyp_ann, collar, ignore_overlap)
                file_metrics["file_id"] = sample_id
                file_metrics["duration"] = entry.get("duration", 0.0)

                if file_metrics["ref_spks"] == file_metrics["hyp_spks"]:
                    correct_spk_count += 1

                per_file.append(file_metrics)
                cumulative_metric(ref_ann, hyp_ann, detailed=True)

            aggregate = self._build_aggregate(cumulative_metric, correct_spk_count, len(eval_manifest))

            eval_result = {
                "aggregate": aggregate,
                "per_file": per_file,
                "eval_settings": {
                    "mode": mode_name,
                    "collar": collar,
                    "ignore_overlap": ignore_overlap,
                    "streaming": streaming,
                },
            }

            all_results[mode_name] = eval_result

            logger.info(
                f"[{mode_name}] DER={aggregate['DER']:.4f} "
                f"(FA={aggregate['FA']:.4f}, Miss={aggregate['Miss']:.4f}, "
                f"Conf={aggregate['Confusion']:.4f}) "
                f"SpkAcc={aggregate['spk_count_acc']:.4f}"
            )

        return all_results

    def _build_aggregate(
        self,
        cumulative_metric: DiarizationErrorRate,
        correct_spk_count: int,
        total_files: int,
    ) -> Dict[str, float]:
        total = cumulative_metric["total"]
        if total > 0:
            return {
                "DER": round(abs(cumulative_metric), 4),
                "FA": round(cumulative_metric["false alarm"] / total, 4),
                "Miss": round(cumulative_metric["missed detection"] / total, 4),
                "Confusion": round(cumulative_metric["confusion"] / total, 4),
                "spk_count_acc": round(correct_spk_count / max(total_files, 1), 4),
                "num_files": total_files,
            }
        return {
            "DER": 0.0,
            "FA": 0.0,
            "Miss": 0.0,
            "Confusion": 0.0,
            "spk_count_acc": 0.0,
            "num_files": total_files,
        }
