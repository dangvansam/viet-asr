import json
import os
from typing import Dict, List, Optional

from loguru import logger

from ..inference.multitask import MultitaskInferenceEngine, SpeakerResult


class MultitaskEvaluator:
    """Evaluates multi-task ASR: WER/CER + paralinguistic task accuracy."""

    def __init__(self, engine: MultitaskInferenceEngine):
        self._engine = engine

    def evaluate(
        self,
        manifest_path: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate on a manifest and return per-task metrics.

        Args:
            manifest_path: JSONL with audio_filepath, text, emotion, gender, age, voice_state
            output_dir: Optional directory for detailed report

        Returns:
            {wer, cer, emotion_acc, gender_acc, age_acc, voice_state_acc}
        """
        entries = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                entries.append(json.loads(line))

        predictions = []
        references = []

        for entry in entries:
            audio_path = entry.get("audio_filepath")
            if not audio_path or not os.path.exists(audio_path):
                logger.warning(f"Skipping missing audio: {audio_path}")
                continue

            results = self._engine.infer(audio_path)
            if not results:
                continue

            pred = results[0]
            predictions.append(pred)
            references.append(entry)

        metrics = {}

        # ASR metrics
        asr_metrics = self._compute_asr_metrics(predictions, references)
        metrics.update(asr_metrics)

        # Per-task accuracy
        for task in ["emotion", "gender", "age", "voice_state"]:
            acc = self._compute_task_accuracy(task, predictions, references)
            if acc is not None:
                metrics[f"{task}_acc"] = acc

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self._generate_report(metrics, predictions, references, output_dir)

        return metrics

    def _compute_asr_metrics(
        self,
        predictions: List[SpeakerResult],
        references: List[dict],
    ) -> Dict[str, float]:
        """Compute WER and CER."""
        total_words = 0
        total_word_errors = 0
        total_chars = 0
        total_char_errors = 0

        for pred, ref in zip(predictions, references):
            ref_text = ref.get("text", "")
            pred_text = pred.text_refined or pred.text

            ref_words = ref_text.split()
            pred_words = pred_text.split()

            word_errors = self._edit_distance(ref_words, pred_words)
            total_words += max(len(ref_words), 1)
            total_word_errors += word_errors

            char_errors = self._edit_distance(list(ref_text), list(pred_text))
            total_chars += max(len(ref_text), 1)
            total_char_errors += char_errors

        wer = total_word_errors / max(total_words, 1)
        cer = total_char_errors / max(total_chars, 1)
        return {"wer": wer, "cer": cer}

    def _compute_task_accuracy(
        self,
        task: str,
        predictions: List[SpeakerResult],
        references: List[dict],
    ) -> Optional[float]:
        """Compute accuracy for a specific paralinguistic task."""
        correct = 0
        total = 0

        for pred, ref in zip(predictions, references):
            ref_label = ref.get(task)
            if ref_label is None:
                continue

            pred_label = getattr(pred, task, None)
            if pred_label is None:
                continue

            total += 1
            if pred_label == ref_label:
                correct += 1

        if total == 0:
            return None
        return correct / total

    def _generate_report(
        self,
        metrics: Dict[str, float],
        predictions: List[SpeakerResult],
        references: List[dict],
        output_dir: str,
    ) -> None:
        """Write evaluation report to output directory."""
        report_path = os.path.join(output_dir, "multitask_eval_report.md")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Multi-Task ASR Evaluation Report\n\n")
            f.write("## Metrics\n\n")
            f.write("| Metric | Value |\n|--------|-------|\n")
            for name, value in sorted(metrics.items()):
                f.write(f"| {name} | {value:.4f} |\n")

            f.write(f"\n## Samples Evaluated: {len(predictions)}\n")

        # Save raw predictions
        preds_path = os.path.join(output_dir, "predictions.jsonl")
        with open(preds_path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred.to_dict(), ensure_ascii=False) + "\n")

        logger.success(f"Evaluation report saved to {report_path}")

    @staticmethod
    def _edit_distance(ref: list, hyp: list) -> int:
        """Compute Levenshtein edit distance."""
        n, m = len(ref), len(hyp)
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, m + 1):
                temp = dp[j]
                if ref[i - 1] == hyp[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return dp[m]
