import json
import os
from typing import Any, Dict, List, Optional

from loguru import logger

from ..configs import EvalConfig
from .evaluator import DiarizationEvaluator
from .metrics.der import DERMetric
from .metrics.latency import LatencyMetric
from .reporters.chart import ChartReporter
from .reporters.text import TextReporter
from .synthesizer import EvalDataSynthesizer


class EvaluationPipeline:
    def __init__(self, cfg: EvalConfig):
        self._cfg = cfg
        self._synthesizer = EvalDataSynthesizer()
        self._evaluator = DiarizationEvaluator(cfg)
        self._der_metric = DERMetric(eval_mode=cfg.eval_mode)
        self._latency_metric = LatencyMetric()
        self._text_reporter = TextReporter(
            model_path=cfg.diar_model_path,
            streaming=cfg.streaming,
        )
        self._chart_reporter = ChartReporter()

    def run(self, eval_manifest: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if eval_manifest is None:
            logger.info("Synthesizing evaluation data...")
            eval_manifest = self._synthesizer.synthesize(self._cfg)

        logger.info("Running diarization inference...")
        eval_manifest = self._evaluator.evaluate(eval_manifest)

        logger.info("Computing metrics...")
        all_results = self._der_metric.compute_all_modes(
            eval_manifest, streaming=self._cfg.streaming
        )

        if self._cfg.streaming:
            for mode_name, result in all_results.items():
                result["latency"] = self._latency_metric.aggregate_latency(eval_manifest)

        self._save_metrics(all_results)

        logger.info("Generating reports...")
        self._text_reporter.generate(all_results, self._cfg.output_dir)

        if self._cfg.generate_charts:
            self._chart_reporter.generate(all_results, self._cfg.output_dir)

        return all_results

    def _save_metrics(self, all_results: Dict[str, Any]) -> None:
        os.makedirs(self._cfg.output_dir, exist_ok=True)
        out_path = os.path.join(self._cfg.output_dir, "metrics.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Metrics saved to {out_path}")
